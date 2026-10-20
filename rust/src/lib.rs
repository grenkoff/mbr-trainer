//! C interface of the push/fold solver, loaded from Python with ctypes (see `mbr/native.py`).

mod solver;

use solver::{Dead, Game, Multiway, Params, N, NN, OUTCOMES, RANKS};
use std::slice;

/// Solve a final-table push/fold game.
///
/// Players are in preflop order, m of them. Inputs (row-major, f64 unless noted):
/// - `w`, `eq`, `tie`: 169x169 card-removal pair weights, all-in equities (a tie counts as half a win)
///   and tie probabilities; `cc`: combos per class;
/// - `v_steal`: m x m, row i = player values when i's shove takes the blinds;
/// - `v_iwin`, `v_jwin`: per pair (i, j) in the order (0,1), (0,2), .., values when i or j wins;
/// - `v_tie`: per pair, the values of a tie minus the mean of `v_iwin` and `v_jwin`;
/// - `v3`: per triple (i, j, k) in the same order, 13 x m values: for each strength order, then for each
///   tie the value of the tie minus the mean value of the orders it fits;
/// - `t3`: 14 pointers to the f32 tensors (169 x 169*169): the 13 outcomes, the orders first, then the
///   combo triple count; and `d_all` (f32, 169*169); or null for no overcalls;
/// - `de`, `dt`: 169*169 x 13 equity and tie changes per dead card of each rank, `ranks`: 169 x 13 cards
///   of each rank per class, `strength`: 169 x 13 per-hand part of `de`; or `de` null to leave the folded
///   players' cards out;
/// - `params`: hu_iterations, tol, mw_iterations, warm_weight, warm_min_freq, mw_min_freq, threads, averaging.
///
/// Outputs: strategies per decision node (shoves, first calls, overcalls, 169 values each), and
/// `ev_play` (169 per node) and `ev_fold` (one per node) in the same node order. Returns 0 on success.
/// With `evaluate_only`, `out_strategy` holds strategies to evaluate once instead of solving.
///
/// # Safety
/// Every pointer must point to an array of the size given above (`t3`, `d_all` and the dead-card
/// arrays may be null).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn mbr_solve(
    m: usize,
    w: *const f64,
    eq: *const f64,
    tie: *const f64,
    cc: *const f64,
    v_steal: *const f64,
    v_iwin: *const f64,
    v_jwin: *const f64,
    v_tie: *const f64,
    v3: *const f64,
    t3: *const *const f32,
    d_all: *const f32,
    de: *const f64,
    dt: *const f64,
    ranks: *const f64,
    strength: *const f64,
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
        let ptrs = slice::from_raw_parts(t3, OUTCOMES + 1);
        let t: [&[f32]; OUTCOMES] = std::array::from_fn(|o| slice::from_raw_parts(ptrs[o], N * NN));
        let mass = slice::from_raw_parts(ptrs[OUTCOMES], N * NN);
        Some(Multiway { t, d_all: slice::from_raw_parts(d_all, NN), mass })
    } else {
        None
    };
    let dead = if de.is_null() {
        None
    } else {
        Some(Dead {
            de: slice::from_raw_parts(de, NN * RANKS),
            dt: slice::from_raw_parts(dt, NN * RANKS),
            ranks: slice::from_raw_parts(ranks, N * RANKS),
            strength: slice::from_raw_parts(strength, N * RANKS),
        })
    };
    let game = Game::new(
        m,
        slice::from_raw_parts(w, NN),
        slice::from_raw_parts(eq, NN),
        slice::from_raw_parts(tie, NN),
        slice::from_raw_parts(cc, N),
        slice::from_raw_parts(v_steal, m * m),
        slice::from_raw_parts(v_iwin, np * m),
        slice::from_raw_parts(v_jwin, np * m),
        slice::from_raw_parts(v_tie, np * m),
        if nt > 0 { slice::from_raw_parts(v3, nt * OUTCOMES * m) } else { &[] },
        mw,
        dead,
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
        let ev = game.evaluate(&s, nt > 0, params.threads, params.mw_min_freq, &mut solver::DeadCache::default());
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
