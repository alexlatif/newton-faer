#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
mod linalg;
mod solver;

pub use linalg::{DenseLu, FaerLu};
pub use solver::{
    Control, IterationStats, Iterations, MatrixFormat, NewtonCfg, solve, solve_cb, solve_dense_cb,
    solve_sparse_cb,
};

use core::fmt::{self, Display, Formatter};
use core::num::NonZeroUsize;
use faer::Mat;
use faer::prelude::SparseColMatRef;
use faer::sparse::SymbolicSparseColMat;
use faer_traits::ComplexField;
use num_traits::Zero;
use std::sync::OnceLock;

pub trait RowMap {
    type Var: Copy + Eq;
    fn dim(&self) -> usize;
    fn row(&self, bus: usize, var: Self::Var) -> Option<usize>;
}

#[derive(Debug, Clone)]
pub struct Pattern<T> {
    pub symbolic: SymbolicSparseColMat<usize>,
    pub values: Vec<T>,
}

impl<T> Pattern<T> {
    #[inline]
    pub fn attach_values(&self) -> SparseColMatRef<'_, usize, T> {
        SparseColMatRef::new(self.symbolic.as_ref(), &self.values)
    }
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

pub trait NonlinearSystem {
    type Real: num_traits::Float;
    type Layout: RowMap;
    type Error;

    fn layout(&self) -> &Self::Layout;
    fn jacobian(&self) -> &dyn JacobianCache<Self::Real>;
    fn jacobian_mut(&mut self) -> &mut dyn JacobianCache<Self::Real>;
    fn residual(&self, x: &[Self::Real], out: &mut [Self::Real]) -> Result<(), Self::Error>;
    fn refresh_jacobian(&mut self, x: &[Self::Real]) -> Result<(), Self::Error>;

    fn jacobian_dense(
        &mut self,
        x: &[Self::Real],
        jac: &mut faer::mat::Mat<Self::Real>,
    ) -> Result<(), Self::Error> {
        self.refresh_jacobian(x)?;
        let sparse = self.jacobian().attach();
        jac.fill(Self::Real::zero());
        let row_idx = sparse.symbolic().row_idx();
        let vals = sparse.val();
        for col in 0..sparse.ncols() {
            let range = sparse.col_range(col);
            for idx in range.clone() {
                jac[(row_idx[idx], col)] = vals[idx];
            }
        }
        Ok(())
    }
}

pub trait LinearSolver<T: ComplexField<Real = T>, M, E> {
    fn factor(&mut self, a: &M) -> SolverResult<(), E>;
    fn solve_in_place(&mut self, rhs: &mut Mat<T>) -> SolverResult<(), E>;
}

pub trait JacobianCache<T /* Real */> {
    fn symbolic(&self) -> &SymbolicSparseColMat<usize>;
    fn values(&self) -> &[T];
    fn values_mut(&mut self) -> &mut [T];
    #[inline]
    fn attach(&self) -> SparseColMatRef<'_, usize, T> {
        SparseColMatRef::new(self.symbolic().as_ref(), self.values())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SolverError<E> {
    /// Problem within the solver internals
    Solver,
    /// Problem in the user's nonlinear system.
    NonLinearSystem(E),
}

impl<E: std::fmt::Display> Display for SolverError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::Solver => f.write_str("solver error"),
            SolverError::NonLinearSystem(err) => err.fmt(f),
        }
    }
}

impl<E> std::error::Error for SolverError<E> where E: std::fmt::Debug + std::fmt::Display {}

pub type SolverResult<T, E> = Result<T, error_stack::Report<SolverError<E>>>;

static RAYON_INIT: OnceLock<usize> = OnceLock::new();

pub fn init_global_parallelism(threads: usize) -> usize {
    if let Some(n) = RAYON_INIT.get().copied() {
        return n;
    }
    let target = if threads == 0 {
        std::thread::available_parallelism()
            .unwrap_or(unsafe { NonZeroUsize::new_unchecked(1) })
            .get()
    } else {
        threads
    };

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(target)
        .build_global();

    let actual = rayon::current_num_threads();
    let _ = RAYON_INIT.set(actual);
    actual
}

#[inline]
pub fn current_parallelism() -> usize {
    RAYON_INIT
        .get()
        .copied()
        .unwrap_or_else(rayon::current_num_threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Pair;
    use faer::sparse::SymbolicSparseColMat;

    #[derive(Clone)]
    struct TwoVarLayout;
    impl RowMap for TwoVarLayout {
        type Var = ();
        fn dim(&self) -> usize {
            2
        }
        fn row(&self, _bus: usize, _var: Self::Var) -> Option<usize> {
            None
        }
    }

    #[derive(Clone)]
    struct Jc {
        sym: SymbolicSparseColMat<usize>,
        vals: Vec<f64>,
    }
    impl JacobianCache<f64> for Jc {
        fn symbolic(&self) -> &SymbolicSparseColMat<usize> {
            &self.sym
        }
        fn values(&self) -> &[f64] {
            &self.vals
        }
        fn values_mut(&mut self) -> &mut [f64] {
            &mut self.vals
        }
    }

    struct Model {
        layout: TwoVarLayout,
        jac: Jc,
    }

    impl Model {
        fn new() -> Self {
            let pairs = vec![
                Pair { row: 0, col: 0 },
                Pair { row: 1, col: 0 },
                Pair { row: 0, col: 1 },
                Pair { row: 1, col: 1 },
            ];
            let (sym, _argsort) = SymbolicSparseColMat::try_new_from_indices(2, 2, &pairs).unwrap();
            let nnz = sym.col_ptr()[sym.ncols()];
            Self {
                layout: TwoVarLayout,
                jac: Jc {
                    sym,
                    vals: vec![0.0; nnz],
                },
            }
        }
    }

    impl NonlinearSystem for Model {
        type Real = f64;
        type Layout = TwoVarLayout;
        type Error = &'static str;

        fn layout(&self) -> &Self::Layout {
            &self.layout
        }

        fn jacobian(&self) -> &dyn JacobianCache<Self::Real> {
            &self.jac
        }
        fn jacobian_mut(&mut self) -> &mut dyn JacobianCache<Self::Real> {
            &mut self.jac
        }

        fn residual(&self, x: &[Self::Real], out: &mut [Self::Real]) -> Result<(), Self::Error> {
            let (xx, yy) = (x[0], x[1]);
            out[0] = xx + yy - 3.0;
            out[1] = xx * xx + yy - 3.0;
            Ok(())
        }

        fn refresh_jacobian(&mut self, x: &[Self::Real]) -> Result<(), Self::Error> {
            let xx = x[0];
            let v = self.jac.values_mut();
            v[0] = 1.0;
            v[1] = 2.0 * xx;
            v[2] = 1.0;
            v[3] = 1.0;
            Ok(())
        }
    }

    #[test]
    fn solves_two_equations_sparse() {
        let cfg = NewtonCfg::<f64>::sparse()
            .with_adaptive(true)
            .with_threads(1);

        let mut model = Model::new();
        let mut x = [0.9_f64, 2.1_f64];

        let iters = crate::solve_sparse_cb(
            &mut model,
            &mut x,
            &mut crate::FaerLu::<f64>::default(),
            cfg,
            |_| Control::Continue,
        )
        .expect("solver");

        assert!(iters > 0 && iters <= 25);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }
}
