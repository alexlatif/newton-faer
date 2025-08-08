# newton-faer

## Why another Newton solver?
Most Rust numerical computing is still young. The existing options:
- roots — great for 1D, but not systems
- argmin — focused on optimization (line searches, trust regions), not sparse Newton for general nonlinear systems
- nalgebra — has solvers, but not a sparse-aware Newton–Raphson with reusable symbolic factorizations

This crate is a thin, reusable Newton core that leans on faer for world-class sparse linear algebra, while keeping all domain logic (residual/Jacobian) outside the engine. You get:
- Separation of concerns: engine = iteration policy + linear solves; models = residual/Jacobian.
- Sparse-first: symbolic LU reused across solves when the sparsity pattern is unchanged.
- Production knobs: adaptive damping, divergence guard + backtracking, cancellation, and progress callbacks.
- Parallelism control: slam all cores for big single cases, or run many small cases in parallel with single-threaded LU (no oversubscription).

Architecture at a glance
- linalg: adapters over faer (sparse LU) and dense LU; keeps symbolic factorization cached per pattern.
- solver: Newton loop, damping policy, divergence guard, callbacks, and thread init.

Your model implements NonlinearSystem (residual + Jacobian/refresh), optionally with its own Jacobian cache.


## Parallelism

Parallelism is a policy choice: inside LU for one/few huge systems, or over cases for large batches. We auto-init Rayon once; you can override the global thread count via config.

```rust
use rayon::prelude::*;
use crate::solver::NewtonCfg;

// batch mode: many cases, maximize throughput
let cfg_batch = NewtonCfg::<f64>::default().with_threads(1); // LU single-thread
let results: Vec<_> = contingencies
    .par_iter()
    .map(|&br| {
        let mut pf = base_pf.clone();
        pf.update_for_outage(br)?;
        pf.solve_with_cfg(cfg_batch) // your wrapper that passes cfg in
    })
    .collect();

// big single case mode: few huge systems
let cfg_big = NewtonCfg::<f64>::default().with_threads(0); // use all cores in LU
let mut pf = big_system_pf.clone();
let res = pf.solve_with_cfg(cfg_big)?;
```
Guideline:
- Tons of scenarios? with_threads(1) + par_iter() over cases.
- One gigantic system? with_threads(0) (all cores) + sequential cases.


## Adaptive Damping

Divergence guard + backtracking are enabled when adaptive=true and steered by:
min_damping, max_damping, grow, shrink, divergence_ratio, ls_backtrack, ls_max_steps.

```rust
let cfg = NewtonCfg::default().with_adaptive(true);
```


## Reuse LU

Keep a solver instance and reuse the symbolic factorization across solves with the same sparsity. Only the numeric phase is recomputed.

```rust
let mut lu = FaerLu::<f64>::default();
for case in cases {
    let mut x = init_state(&case);
    let _iters = solver::solve_sparse_cb(&mut model, &mut x, &mut lu, cfg, |_| Control::Continue)?;
}
```

## Solver Progress

Stream iteration stats to your UI, support cancellation, and run the solve on a worker thread.

```rust
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, mpsc,
};
use crate::solver::{Control, IterationStats};

let (tx, rx) = mpsc::channel::<IterationStats<f64>>();
let cancel = Arc::new(AtomicBool::new(false));
let cancel_flag = cancel.clone();

let mut system = /* build system */;
std::thread::spawn(move || {
    let _ = solver::solve_cb(&mut model, &mut x, cfg.with_adaptive(true), |st| {
        let _ = tx.send(st.clone());
        if cancel_flag.load(Ordering::Relaxed) { Control::Cancel } else { Control::Continue }
    });
});

while let Ok(st) = rx.recv() {
    println!("iter={} residual={:.3e} damping={:.2}", st.iter, st.residual, st.damping);
    // if user hits "Stop": cancel.store(true, Ordering::Relaxed);
}
```

## Performance notes
- Symbolic reuse is the big win in multi-scenario studies (fixed sparsity).
- Preallocated buffers and in-place solves avoid per-iteration allocations.
- Threading: pick one level of parallelism—inside LU or over cases—to avoid oversubscription.
- Ordering (AMD/ND/…) can drastically reduce fill; we expose knobs for faer’s symbolic parameters if you want to tune.