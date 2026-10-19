//! C interface of the push/fold solver, loaded from Python with ctypes (see `mbr/native.py`).

mod solver;

use solver::{Game, Multiway, Params, N, NN, ORDERS};
use std::slice;

/// Solve a final-table push/fold game.
///
/// Players are in preflop order, m of them. Inputs (row-major, f64 unless noted):
/// - `w`, `eq`: 169x169 card-removal pair weights and all-in equities; `cc`: combos per class;
/// - `v_steal`: m x m, row i = player values when i's shove takes the blinds;
/// - `v_iwin`, `v_jwin`: per pair (i, j) in the order (0,1), (0,2), .., values when i or j wins;
/// - `v3`: per triple (i, j, k) in the same order, 6 x m values for each strength order;
/// - `t3`: 6 pointers to the f32 order tensors (169 x 169*169) and `d_all` (f32, 169*169), or null
///   for no overcalls;
/// - `params`: hu_iterations, tol, mw_iterations, warm_weight, warm_min_freq, mw_min_freq, threads, averaging.
///
/// Outputs: strategies per decision node (shoves, first calls, overcalls, 169 values each), and
/// `ev_play` (169 per node) and `ev_fold` (one per node) in the same node order. Returns 0 on success.
/// With `evaluate_only`, `out_strategy` holds strategies to evaluate once instead of solving.
///
/// # Safety
/// Every pointer must point to an array of the size given above (`t3` and `d_all` may be null).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mbr_solve(
    m: usize,
    w: *const f64,
    eq: *const f64,
    cc: *const f64,
    v_steal: *const f64,
    v_iwin: *const f64,
    v_jwin: *const f64,
    v3: *const f64,
    t3: *const *const f32,
    d_all: *const f32,
    params: *const f64,
    out_strategy: *mut f64,
    out_ev_play: *mut f64,
    out_ev_fold: *mut f64,
    evaluate_only: i32,
) -> i32 {
    if !(2..=9).contains(&m) {
        return 1;
    }
    let np = m * (m - 1) / 2;
    let multiway = !t3.is_null() && !d_all.is_null() && m >= 3;
    let nt = if multiway { m * (m - 1) * (m - 2) / 6 } else { 0 };
    let mw = if multiway {
        let ptrs = slice::from_raw_parts(t3, ORDERS);
        let t: [&[f32]; ORDERS] = std::array::from_fn(|o| slice::from_raw_parts(ptrs[o], N * NN));
        Some(Multiway { t, d_all: slice::from_raw_parts(d_all, NN) })
    } else {
        None
    };
    let game = Game::new(
        m,
        slice::from_raw_parts(w, NN),
        slice::from_raw_parts(eq, NN),
        slice::from_raw_parts(cc, N),
        slice::from_raw_parts(v_steal, m * m),
        slice::from_raw_parts(v_iwin, np * m),
        slice::from_raw_parts(v_jwin, np * m),
        if nt > 0 { slice::from_raw_parts(v3, nt * ORDERS * m) } else { &[] },
        mw,
    );
    let pr = slice::from_raw_parts(params, 8);
    let threads = if pr[6] >= 1.0 {
        pr[6] as usize
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    };
    let params = Params {
        hu_iterations: pr[0] as usize,
        tol: pr[1],
        mw_iterations: pr[2] as usize,
        warm_weight: pr[3],
        warm_min_freq: pr[4],
        mw_min_freq: pr[5],
        threads,
        averaging: pr[7],
    };
    let (s, ev) = if evaluate_only != 0 {
        // the strategies come in through out_strategy; evaluate them once
        let strategy = slice::from_raw_parts(out_strategy, ((m - 1) + np + nt) * N);
        let rows = |from: usize, count: usize| -> Vec<Vec<f64>> {
            (0..count).map(|r| strategy[(from + r) * N..(from + r + 1) * N].to_vec()).collect()
        };
        let s = solver::Strategies { push: rows(0, m - 1), call: rows(m - 1, np), over: rows(m - 1 + np, nt) };
        let ev = game.evaluate(&s, nt > 0, params.threads, params.mw_min_freq);
        (s, ev)
    } else {
        solver::solve(&game, &params)
    };

    let nodes = (m - 1) + np + nt;
    let strategy = slice::from_raw_parts_mut(out_strategy, nodes * N);
    let ev_play = slice::from_raw_parts_mut(out_ev_play, nodes * N);
    let ev_fold = slice::from_raw_parts_mut(out_ev_fold, nodes);
    let strategies = s.push.iter().chain(&s.call).chain(&s.over);
    let evs = ev.push.iter().chain(&ev.call).chain(&ev.over);
    for (node, (x, e)) in strategies.zip(evs).enumerate() {
        strategy[node * N..(node + 1) * N].copy_from_slice(x);
        ev_play[node * N..(node + 1) * N].copy_from_slice(&e.0);
        ev_fold[node] = e.1;
    }
    0
}
