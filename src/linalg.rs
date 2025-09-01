use super::{ComplexField, LinearSolver, Mat, SolverError, SolverResult};
use dyn_stack::{MemBuffer, MemStack};
use error_stack::Report;
use error_stack::ResultExt;
use faer::linalg::solvers::ShapeCore;
use faer::{
    Conj, Par,
    linalg::solvers::FullPivLu,
    mat::{MatMut, MatRef},
    prelude::{Solve, SolveLstsq},
    sparse::{
        SparseColMatRef,
        linalg::lu::{LuRef, LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu},
        linalg::solvers::{Qr, SymbolicQr},
    },
};

#[inline]
fn fnv1a64_init() -> u64 {
    0xcbf29ce484222325
}
#[inline]
fn fnv1a64_step(mut h: u64, v: u64) -> u64 {
    h ^= v;
    h = h.wrapping_mul(0x100000001b3);
    h
}
#[inline]
fn hash_usize_slice(mut h: u64, s: &[usize]) -> u64 {
    for &x in s {
        #[cfg(target_pointer_width = "64")]
        {
            h = fnv1a64_step(h, x as u64);
        }
        #[cfg(target_pointer_width = "32")]
        {
            h = fnv1a64_step(h, x as u64);
        }
    }
    h
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PatternSig {
    nrows: usize,
    ncols: usize,
    nnz: usize,
    col_ptr_hash: u64,
    row_idx_hash: u64,
    col_ptr_ptr: *const usize,
    row_idx_ptr: *const usize,
}

fn pattern_sig<T>(a: &SparseColMatRef<'_, usize, T>) -> PatternSig {
    let sym = a.symbolic();
    let col_ptr = sym.col_ptr();
    let row_idx = sym.row_idx();

    let col_ptr_ptr = col_ptr.as_ptr();
    let row_idx_ptr = row_idx.as_ptr();

    let mut h = fnv1a64_init();
    let col_ptr_hash = hash_usize_slice(h, col_ptr);
    h = fnv1a64_init();
    let row_idx_hash = hash_usize_slice(h, row_idx);

    PatternSig {
        nrows: a.nrows(),
        ncols: a.ncols(),
        nnz: *col_ptr.last().unwrap_or(&0),
        col_ptr_hash,
        row_idx_hash,
        col_ptr_ptr,
        row_idx_ptr,
    }
}

pub struct FaerLu<T: ComplexField<Real = T>> {
    sym: Option<SymbolicLu<usize>>,
    num: NumericLu<usize, T>,
    scratch: Option<MemBuffer>,
    // Don’t share one FaerLu across threads.
    // It’s a mutable solver with internal scratch;
    // instead create one solver per worker and still reuse within that worker across many solves.
    sig: Option<PatternSig>,
}

impl<T: ComplexField<Real = T>> Default for FaerLu<T> {
    fn default() -> Self {
        Self {
            sym: None,
            num: NumericLu::new(),
            scratch: None,
            sig: None,
        }
    }
}

impl<T: ComplexField<Real = T>> LinearSolver<T, SparseColMatRef<'_, usize, T>> for FaerLu<T> {
    fn factor(&mut self, a: &SparseColMatRef<'_, usize, T>) -> SolverResult<()> {
        let now = pattern_sig(a);
        let par = Par::rayon(0);

        let need_symbolic = match self.sig {
            None => true,
            Some(prev) => {
                if prev.col_ptr_ptr == now.col_ptr_ptr && prev.row_idx_ptr == now.row_idx_ptr {
                    false
                } else {
                    prev != now
                }
            }
        };

        if need_symbolic {
            self.sym = Some(
                factorize_symbolic_lu(a.symbolic(), LuSymbolicParams::default())
                    .attach_printable("LU symbolic factorization failed")
                    .change_context(SolverError)?,
            );

            let scratch_size = self
                .sym
                .as_ref()
                .ok_or(SolverError)
                .attach_printable("Symbolic factorization missing")?
                .factorize_numeric_lu_scratch::<T>(par, Default::default());
            self.scratch = Some(MemBuffer::new(scratch_size));
            self.sig = Some(now);
        }

        let mut stack = MemStack::new(
            self.scratch
                .as_mut()
                .ok_or(SolverError)
                .attach_printable("Scratch buffer not initialized")?,
        );

        self.sym
            .as_ref()
            .ok_or(SolverError)
            .attach_printable("Symbolic factorization not available")?
            .factorize_numeric_lu(&mut self.num, *a, par, &mut stack, Default::default())
            .attach_printable("Numeric LU factorization failed")
            .change_context(SolverError)?;

        Ok(())
    }

    fn solve_into(&mut self, rhs: MatRef<T>, mut out: MatMut<T>) -> SolverResult<()> {
        let mut stack = MemStack::new(
            self.scratch
                .as_mut()
                .ok_or(SolverError)
                .attach_printable("Scratch buffer not available for solve")?,
        );

        let lu_ref = unsafe {
            LuRef::new_unchecked(
                self.sym
                    .as_ref()
                    .ok_or(SolverError)
                    .attach_printable("Symbolic factorization not available for solve")?,
                &self.num,
            )
        };

        // Since the underlying solver is in-place, we first copy the rhs data
        // into the output buffer.
        out.copy_from(rhs);

        // Then we solve in-place, modifying `out`` to contain the solution.
        lu_ref.solve_in_place_with_conj(Conj::No, out.as_mut(), Par::rayon(0), &mut stack);

        Ok(())
    }
}

pub struct SparseQr<T: ComplexField<Real = T>> {
    symbolic: Option<SymbolicQr<usize>>,
    qr: Option<Qr<usize, T>>,
    sig: Option<PatternSig>,
}

impl<T: ComplexField<Real = T>> Default for SparseQr<T> {
    fn default() -> Self {
        Self {
            symbolic: None,
            qr: None,
            sig: None,
        }
    }
}

impl<T: ComplexField<Real = T>> LinearSolver<T, SparseColMatRef<'_, usize, T>> for SparseQr<T> {
    fn factor(&mut self, a: &SparseColMatRef<'_, usize, T>) -> SolverResult<()> {
        let now = pattern_sig(a);

        let need_symbolic = match self.sig {
            None => true,
            Some(prev) => {
                if prev.col_ptr_ptr == now.col_ptr_ptr && prev.row_idx_ptr == now.row_idx_ptr {
                    false
                } else {
                    prev != now
                }
            }
        };

        if need_symbolic {
            self.symbolic = Some(
                SymbolicQr::try_new(a.symbolic())
                    .attach_printable("QR symbolic factorization failed")
                    .change_context(SolverError)?,
            );
            self.sig = Some(now);
        }

        // Create the numeric QR factorization
        self.qr = Some(
            Qr::try_new_with_symbolic(
                self.symbolic
                    .as_ref()
                    .ok_or(SolverError)
                    .attach_printable("Symbolic factorization not available")?
                    .clone(),
                *a,
            )
            .attach_printable("Numeric QR factorization failed")
            .change_context(SolverError)?,
        );

        Ok(())
    }

    fn solve_into(&mut self, rhs: MatRef<T>, mut out: MatMut<T>) -> SolverResult<()> {
        let qr = self
            .qr
            .as_ref()
            .ok_or(SolverError)
            .attach_printable("QR factorization not available for solve")?;

        // For QR least squares, the input and output dimensions may be different.
        // rhs is n_residuals × n_rhs_cols
        // out: should be able to hold at least n_variables × n_rhs_cols;
        //      QR expects to work with an n_residuals × n_rhs_cols buffer

        // Check dimensions are compatible.
        if out.nrows() < qr.ncols() || out.ncols() != rhs.ncols() {
            return Err(Report::new(SolverError)
                .attach_printable("Output buffer too small for QR solution"));
        }

        // We need to work in a larger buffer and copy the result back.
        if out.nrows() == rhs.nrows() {
            // Square system; can work directly.
            out.copy_from(rhs);
            qr.solve_lstsq_in_place(out.as_mut());
        } else {
            // Non-square system; use temporary buffer.
            let mut work = faer::mat::Mat::zeros(rhs.nrows(), rhs.ncols());
            work.copy_from(rhs);
            qr.solve_lstsq_in_place(work.as_mut());

            // Copy solution (first n_variables rows) back to output.
            for j in 0..out.ncols() {
                for i in 0..out.nrows() {
                    out[(i, j)] = work[(i, j)].clone();
                }
            }
        }

        Ok(())
    }
}

pub struct DenseLu<T: ComplexField<Real = T>> {
    lu: Option<FullPivLu<T>>,
}

impl<T: ComplexField<Real = T>> Default for DenseLu<T> {
    fn default() -> Self {
        Self { lu: None }
    }
}

impl<T: ComplexField<Real = T>> LinearSolver<T, Mat<T>> for DenseLu<T> {
    fn factor(&mut self, a: &Mat<T>) -> SolverResult<()> {
        self.lu = Some(a.full_piv_lu());
        Ok(())
    }

    // This is the updated method
    fn solve_into(&mut self, rhs: MatRef<T>, mut out: MatMut<T>) -> SolverResult<()> {
        let lu = self
            .lu
            .as_ref()
            .ok_or(SolverError)
            .attach_printable("Dense LU not factorized")?;

        // `lu.solve` returns a new matrix with the solution.
        let solution = lu.solve(rhs);

        // We copy the solution into the provided output buffer.
        out.copy_from(&solution);

        Ok(())
    }
}
