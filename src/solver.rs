use super::{
    LinearSolver, Mat, NonlinearSystem, RowMap, SolverError, SolverResult, SparseColMatRef,
    init_global_parallelism,
    linalg::{DenseLu, FaerLu},
};
use error_stack::Report;
use faer::mat::Mat as FaerMat;
use faer_traits::ComplexField;
use num_traits::{Float, One, ToPrimitive, Zero};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatrixFormat {
    Sparse,
    Dense,
    Auto(usize),
}

impl Default for MatrixFormat {
    fn default() -> Self {
        Self::Auto(100)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NewtonCfg<T> {
    pub tol: T,
    pub damping: T,
    pub max_iter: usize,
    pub format: MatrixFormat,

    // step control
    pub adaptive: bool,
    pub min_damping: T,
    pub max_damping: T,
    pub grow: T,
    pub shrink: T,
    pub divergence_ratio: T,
    pub ls_backtrack: T,
    pub ls_max_steps: usize,

    pub n_threads: usize,
}

impl<T: Float> Default for NewtonCfg<T> {
    fn default() -> Self {
        let _ = init_global_parallelism(0);
        Self {
            tol: T::from(1e-8).expect("Type must support 1e-8 for default tolerance"),
            damping: T::one(),
            max_iter: 25,
            format: MatrixFormat::default(),
            adaptive: false,
            min_damping: T::from(0.1).unwrap(),
            max_damping: T::one(),
            grow: T::from(1.1).unwrap(),
            shrink: T::from(0.5).unwrap(),
            divergence_ratio: T::from(3.0).unwrap(),
            ls_backtrack: T::from(0.5).unwrap(),
            ls_max_steps: 10,
            n_threads: 0,
        }
    }
}

impl<T: Float> NewtonCfg<T> {
    pub fn sparse() -> Self {
        Self {
            format: MatrixFormat::Sparse,
            ..Default::default()
        }
    }
    pub fn dense() -> Self {
        Self {
            format: MatrixFormat::Dense,
            ..Default::default()
        }
    }
    pub fn with_format(mut self, format: MatrixFormat) -> Self {
        self.format = format;
        self
    }
    pub fn with_adaptive(mut self, enabled: bool) -> Self {
        self.adaptive = enabled;
        self
    }
    pub fn with_threads(mut self, n_threads: usize) -> Self {
        init_global_parallelism(n_threads);
        self.n_threads = n_threads;
        self
    }
}

pub type Iterations = usize;

#[derive(Clone, Debug)]
pub struct IterationStats<T> {
    pub iter: usize,
    pub residual: T,
    pub damping: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Cancel,
}

fn newton_iterate<M, F, Cb>(
    model: &mut M,
    x: &mut [M::Real],
    cfg: super::NewtonCfg<M::Real>,
    mut solve_into: F,
    mut on_iter: Cb,
) -> SolverResult<Iterations, M::Error>
where
    M: NonlinearSystem,
    M::Real: ComplexField<Real = M::Real> + Float + Zero + One + ToPrimitive,
    M::Error: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static,
    F: FnMut(&mut M, &[M::Real], &[M::Real], &mut [M::Real]) -> SolverResult<(), M::Error>,
    Cb: FnMut(&IterationStats<M::Real>) -> Control,
{
    let n = model.layout().dim();
    let mut f = vec![M::Real::zero(); n];
    let mut dx = vec![M::Real::zero(); n];
    let mut damping = cfg.damping;
    let mut last_res = M::Real::infinity();

    // buffers for line search
    let mut x_trial = vec![M::Real::zero(); n];
    let mut f_trial = vec![M::Real::zero(); n];

    for iter in 0..cfg.max_iter {
        model
            .residual(x, &mut f)
            .map_err(SolverError::NonLinearSystem)?;
        let res = f
            .iter()
            .map(|&v| v.abs())
            .fold(M::Real::zero(), |a, b| if a > b { a } else { b });

        if matches!(
            on_iter(&IterationStats {
                iter,
                residual: res,
                damping
            }),
            Control::Cancel
        ) {
            return Err(Report::new(SolverError::Solver).attach_printable("solve cancelled"));
        }
        if res < cfg.tol {
            return Ok(iter + 1);
        }

        solve_into(model, x, &f, &mut dx)?;

        let mut step_applied = false;

        if cfg.adaptive {
            if res < last_res {
                let nd = damping * cfg.grow;
                damping = if nd > cfg.max_damping {
                    cfg.max_damping
                } else {
                    nd
                };
            } else {
                let nd = damping * cfg.shrink;
                damping = if nd < cfg.min_damping {
                    cfg.min_damping
                } else {
                    nd
                };
            }

            if last_res.is_finite() && res > last_res * cfg.divergence_ratio {
                let mut alpha = if damping * cfg.shrink < cfg.min_damping {
                    cfg.min_damping
                } else {
                    damping * cfg.shrink
                };

                for _ in 0..cfg.ls_max_steps {
                    for i in 0..n {
                        x_trial[i] = x[i] + alpha * dx[i];
                    }
                    model
                        .residual(&x_trial, &mut f_trial)
                        .map_err(SolverError::NonLinearSystem)?;
                    let res_try = f_trial
                        .iter()
                        .map(|&v| v.abs())
                        .fold(M::Real::zero(), |a, b| if a > b { a } else { b });

                    if res_try < res {
                        x.copy_from_slice(&x_trial);
                        damping = alpha;
                        step_applied = true;
                        break;
                    }
                    alpha = alpha * cfg.ls_backtrack;
                    if alpha < cfg.min_damping {
                        break;
                    }
                }

                if !step_applied {
                    return Err(Report::new(SolverError::Solver)
                        .attach_printable("divergence guard: line search failed"));
                }
            }
        }

        if !step_applied {
            for (xi, &dxi) in x.iter_mut().zip(dx.iter()) {
                *xi = *xi + damping * dxi;
            }
        }

        last_res = res;
    }

    Err(Report::new(SolverError::Solver).attach_printable(format!(
        "Newton solver did not converge after {} iterations",
        cfg.max_iter
    )))
}

pub fn solve<M>(
    model: &mut M,
    x: &mut [M::Real],
    cfg: super::NewtonCfg<M::Real>,
) -> SolverResult<Iterations, M::Error>
where
    M: NonlinearSystem,
    M::Real: ComplexField<Real = M::Real> + Float + Zero + One + ToPrimitive,
    M::Error: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
{
    solve_cb(model, x, cfg, |_| Control::Continue)
}

pub fn solve_cb<M, Cb>(
    model: &mut M,
    x: &mut [M::Real],
    cfg: super::NewtonCfg<M::Real>,
    on_iter: Cb,
) -> SolverResult<Iterations, M::Error>
where
    M: NonlinearSystem,
    M::Error: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    M::Real: ComplexField<Real = M::Real> + Float + Zero + One + ToPrimitive,
    Cb: FnMut(&IterationStats<M::Real>) -> Control,
{
    let n = model.layout().dim();
    let use_dense = match cfg.format {
        super::MatrixFormat::Dense => true,
        super::MatrixFormat::Sparse => false,
        super::MatrixFormat::Auto(threshold) => n < threshold,
    };

    if use_dense {
        let mut lu = DenseLu::<M::Real>::default();
        solve_dense_cb(model, x, &mut lu, cfg, on_iter)
    } else {
        let mut lu = FaerLu::<M::Real>::default();
        solve_sparse_cb(model, x, &mut lu, cfg, on_iter)
    }
}

pub fn solve_sparse_cb<M, L, Cb>(
    model: &mut M,
    x: &mut [M::Real],
    lin: &mut L,
    cfg: super::NewtonCfg<M::Real>,
    on_iter: Cb,
) -> SolverResult<Iterations, M::Error>
where
    M: NonlinearSystem,
    L: for<'a> LinearSolver<M::Real, SparseColMatRef<'a, usize, M::Real>, M::Error>,
    M::Real: ComplexField<Real = M::Real> + Float + Zero + One + ToPrimitive,
    M::Error: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    Cb: FnMut(&IterationStats<M::Real>) -> Control,
{
    let n = model.layout().dim();
    let mut rhs = FaerMat::<M::Real>::zeros(n, 1);

    newton_iterate(
        model,
        x,
        cfg,
        |model, x, f, dx| {
            model
                .refresh_jacobian(x)
                .map_err(SolverError::NonLinearSystem)?;
            lin.factor(&model.jacobian().attach())?;

            rhs.col_mut(0)
                .as_mut()
                .iter_mut()
                .zip(f.iter())
                .for_each(|(dst, &src)| *dst = -src);

            lin.solve_in_place(&mut rhs)?;

            for i in 0..n {
                dx[i] = rhs[(i, 0)];
            }
            Ok(())
        },
        on_iter,
    )
}

pub fn solve_dense_cb<M, L, Cb>(
    model: &mut M,
    x: &mut [M::Real],
    lu: &mut L,
    cfg: super::NewtonCfg<M::Real>,
    on_iter: Cb,
) -> SolverResult<Iterations, M::Error>
where
    M: NonlinearSystem,
    L: LinearSolver<M::Real, Mat<M::Real>, M::Error>,
    M::Real: ComplexField<Real = M::Real> + Float + Zero + One + ToPrimitive,
    M::Error: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    Cb: FnMut(&IterationStats<M::Real>) -> Control,
{
    let n = model.layout().dim();
    let mut jac = FaerMat::<M::Real>::zeros(n, n);
    let mut rhs = FaerMat::<M::Real>::zeros(n, 1);

    newton_iterate(
        model,
        x,
        cfg,
        |model, x, f, dx| {
            model
                .jacobian_dense(x, &mut jac)
                .map_err(|e| Report::new(SolverError::NonLinearSystem(e)))?;
            lu.factor(&jac)?;
            for (i, &fi) in f.iter().enumerate() {
                rhs[(i, 0)] = -fi;
            }
            lu.solve_in_place(&mut rhs)?;
            for i in 0..n {
                dx[i] = rhs[(i, 0)];
            }
            Ok(())
        },
        on_iter,
    )
}
