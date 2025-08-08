mod engine;
mod linalg;

pub use engine::{
    Control, IterationStats, Iterations, MatrixFormat, NewtonCfg, solve, solve_cb, solve_dense_cb,
    solve_sparse_cb,
};
pub use linalg::{DenseLu, FaerLu};

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

    fn layout(&self) -> &Self::Layout;
    fn jacobian(&self) -> &dyn JacobianCache<Self::Real>;
    fn jacobian_mut(&mut self) -> &mut dyn JacobianCache<Self::Real>;
    fn residual(&self, x: &[Self::Real], out: &mut [Self::Real]);
    fn refresh_jacobian(&mut self, x: &[Self::Real]);

    fn jacobian_dense(&mut self, x: &[Self::Real], jac: &mut faer::mat::Mat<Self::Real>) {
        self.refresh_jacobian(x);
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
    }
}

pub trait LinearSolver<T: ComplexField<Real = T>, M> {
    fn factor(&mut self, a: &M) -> SolverResult<()>;
    fn solve_in_place(&mut self, rhs: &mut Mat<T>) -> SolverResult<()>;
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
pub struct SolverError;

impl Display for SolverError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("solver error")
    }
}

impl std::error::Error for SolverError {}

pub type SolverResult<T> = Result<T, error_stack::Report<SolverError>>;

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
