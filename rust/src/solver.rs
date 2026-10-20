#![allow(clippy::needless_range_loop)]

//! Fictitious play for the push/fold game, a port of `mbr.pushfold.Solver` (see its docs for the model).
//!
//! Hands are the 169 classes; the grid over (shover hand a, caller hand b) is stored row-major,
//! index `a * N + b`. The multiway part works through the grid a few rows a at a time, so that what it
//! touches stays in cache, and spreads the blocks of rows over threads. Its two contractions with the
//! 3-way tensors are small matrix products (`gemm`): pass A sums the overcalling ranges over the
//! overcaller's hand, pass B sums the shove-and-call lines over the shover's and caller's hands.

use std::thread;
use std::time::Instant;

pub const N: usize = 169;
pub const NN: usize = N * N;
/// Strict strength orders of the three all-in hands: their tensors add up to the combo triple count.
pub const ORDERS: usize = 6;
/// The orders and then the ties (`mbr.equity3.TIES`), whose tensors are corrections to the orders.
pub const OUTCOMES: usize = 13;
/// Per triple, the sums over the outcome tensors and then one over the combo triple count W3 (`Multiway.mass`)
/// for the dead cards in 3-way all-ins.
const SLOTS: usize = OUTCOMES + 1;
/// Card ranks.
pub const RANKS: usize = 13;
/// Two-card combos in a deck.
const COMBOS: f64 = 1326.0;
/// The equities moved by dead cards are rebuilt when an expected rank count moved by more than this
/// since the last build (`DeadCache`): that moves them by 3e-4 at most, and the final EVs are rebuilt.
const DEAD_TOL: f64 = 1e-3;
/// C[o][p] of `mbr.equity3.pair_shift_orders`: how strict order o moves when the first role of pair p
/// (shover-caller, shover-overcaller, caller-overcaller) ends above the second more often by 1.
const PAIR_SHIFT: [[f64; 3]; ORDERS] =
    [[0.5, 0.0, 0.5], [0.0, 0.5, -0.5], [-0.5, 0.5, 0.0], [0.0, -0.5, 0.5], [0.5, -0.5, 0.0], [-0.5, 0.0, -0.5]];

pub struct Params {
    pub hu_iterations: usize,
    pub tol: f64,
    pub mw_iterations: usize,
    pub warm_weight: f64,
    pub warm_min_freq: f64,
    pub mw_min_freq: f64,
    pub threads: usize,
    /// iteration t enters the averaged strategies with weight ~ t^(averaging - 1)
    pub averaging: f64,
    /// the first player is all-in from posting: he always shoves, the others can only call him
    pub forced_open: bool,
}

/// The 3-way outcome tensors: `t[o][c * NN + a * N + b]` = combo weight * P(outcome o) for shover
/// hand a, caller hand b and overcaller hand c; `d_all[a * N + b]` = sum over c and the strict orders.
pub struct Multiway<'a> {
    pub t: [&'a [f32]; OUTCOMES],
    pub d_all: &'a [f32],
    /// the combo triple count W3, the sum of the six order tensors (the same in every axis order)
    pub mass: &'a [f32],
}

/// The cards of the players who folded (`mbr.ranges.dead_card_effects`, `rank_counts`,
/// `dead_card_strength`), all [x][r] row-major with r over the 13 ranks.
pub struct Dead<'a> {
    pub de: &'a [f64],       // [a * N + b][r] equity change per dead card of rank r
    pub dt: &'a [f64],       // the same for the tie probability
    pub ranks: &'a [f64],    // [c][r] cards of rank r in class c
    pub strength: &'a [f64], // [c][r] per-hand part of de, for 3-way all-ins
}

pub struct Game<'a> {
    pub m: usize,
    w: &'a [f64],
    eq: &'a [f64],
    tie: &'a [f64],
    wsum: Vec<f64>,
    w32: Vec<f32>,
    dead: Option<Dead<'a>>,
    dead_by_rank: Option<(Vec<f64>, Vec<f64>)>, // de and dt laid out [r][a * N + b]
    cc: &'a [f64],
    v_steal: Vec<Vec<f64>>,
    pub pairs: Vec<(usize, usize)>,
    v_iwin: Vec<Vec<f64>>,
    v_jwin: Vec<Vec<f64>>,
    v_tie: Vec<Vec<f64>>, // per pair: a tie minus the mean of a win and a loss (eq counts a tie as half a win)
    pub triples: Vec<(usize, usize, usize)>,
    pair_triples: Vec<Vec<usize>>, // for each pair (i, j): the triples (i, j, k) for k = j+1 .. m-1
    triple_pair: Vec<usize>,       // the pair (i, j) of each triple (i, j, k)
    v3: Vec<Vec<f32>>,             // per triple: v3[o * m + x] = value of player x for strength order o
    // per triple: alpha[role * m + x], how the dead cards move player x's value per unit of the
    // strength of the shover's (role 0), caller's (1) and overcaller's (2) hand (PAIR_SHIFT)
    alpha: Vec<Vec<f32>>,
    mw: Option<Multiway<'a>>,
    minor: [usize; OUTCOMES], // the tensor that read as [a][b][c] is U_o, the overcaller's hand innermost
}

/// Grid rows handled together in pass A, and the padded width of their columns.
const ROW_BLOCK: usize = 4;
const BLOCK_COLS: usize = (ROW_BLOCK * N).div_ceil(STRIP) * STRIP;
/// N padded for `gemm`.
const PAD_N: usize = N.div_ceil(STRIP) * STRIP;

/// Place of the hand of each role (shover, caller, overcaller) in every outcome, 0 = best, tied hands
/// sharing a place: the orders of `mbr.equity3.ORDERS`, then the ties of `mbr.equity3.TIES`.
const OUTCOME_PLACES: [[usize; 3]; OUTCOMES] = [
    [0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0],
    [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0],
];

/// Tensor o holds the weights of outcome o with the axes ordered (overcaller, shover, caller). Reading
/// it with the axes (shover, caller, overcaller) gives the outcome where the roles have moved one place:
/// the shover's place goes to the overcaller, the caller's to the shover, the overcaller's to the
/// caller. So U_o with the axes (shover, caller, overcaller) is tensor `minor[o]`.
fn minor_layouts() -> [usize; OUTCOMES] {
    std::array::from_fn(|o| {
        let [s, c, v] = OUTCOME_PLACES[o];
        (0..OUTCOMES).find(|&q| OUTCOME_PLACES[q] == [c, v, s]).unwrap()
    })
}

#[derive(Clone)]
pub struct Strategies {
    pub push: Vec<Vec<f64>>,
    pub call: Vec<Vec<f64>>,
    pub over: Vec<Vec<f64>>,
}

/// Who folded before each decision (`mbr.pushfold.Solver._bunching`). Per pair (i, j), i shoving and j
/// calling, given the players before i and between i and j folded: phi of the hands (the pair weights
/// are W[a, b] phi[a] phi[b]), the row sums of the pair weights, and the expected dead cards of each rank
/// (the equities they move are in the `DeadCache`). With overcalls also per triple the overcaller's phi
/// and how the dead cards move each hand.
struct Bunch {
    pair_phi: Vec<Vec<f64>>,
    pair_wsum: Vec<Vec<f64>>,
    over_phi: Vec<Vec<f64>>,
    strength: Vec<Vec<f64>>,
    opener: Vec<Vec<f64>>, // phi of i's hands when it is folded to i
}

/// Expected cards of each rank and phi of one folding range (zero and one when it never folds).
type FoldStats = (Vec<f64>, Vec<f64>);

/// The equities and ties moved by dead cards, per pair, with the expected rank counts they were built
/// for: [pair * 2 + 0/1] equities/ties given the folds before the caller, and in f32 at the showdown
/// once everybody behind the caller folded too. The folds change little from one iteration to the next.
#[derive(Default)]
pub struct DeadCache {
    slots: Vec<Option<(Vec<f64>, Vec<f64>)>>,
    show: Vec<Option<(Vec<f64>, Vec<f32>)>>,
}

/// Sums over the pair weights W[a, b] phi[a] phi[b] for the first-caller node of a pair: per shover hand a
/// the row sums and the sums times call[b], times call[b] E[a, b], times call[b] T[a, b]; per caller hand b
/// the sums times push[a], times push[a] E[b, a], times push[a] T[a, b].
struct PairSums {
    wsum: Vec<f64>,
    wc: Vec<f64>,
    wec: Vec<f64>,
    wtc: Vec<f64>,
    wp: Vec<f64>,
    wep: Vec<f64>,
    wtp: Vec<f64>,
}

/// EV of playing (shove / call / overcall) for every hand and the EV of folding, per decision node.
pub struct Ev {
    pub push: Vec<(Vec<f64>, f64)>,
    pub call: Vec<(Vec<f64>, f64)>,
    pub over: Vec<(Vec<f64>, f64)>,
}

/// Dot product with independent partial sums, so that it vectorizes.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (xc, yc) = (x.chunks_exact(8), y.chunks_exact(8));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        for l in 0..8 {
            acc[l] = a[l].mul_add(b[l], acc[l]);
        }
    }
    acc.iter().sum::<f64>() + xr.iter().zip(yr).map(|(a, b)| a * b).sum::<f64>()
}

/// f32 dot product summed in f64.
fn dot32(x: &[f32], y: &[f32]) -> f64 {
    let mut acc = [0.0f32; 16];
    let (xc, yc) = (x.chunks_exact(16), y.chunks_exact(16));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        for l in 0..16 {
            acc[l] = a[l].mul_add(b[l], acc[l]);
        }
    }
    acc.iter().map(|&v| v as f64).sum::<f64>() + xr.iter().zip(yr).map(|(a, b)| (a * b) as f64).sum::<f64>()
}

/// Rows of a sparse matrix product, packed four at a time for `gemm`: each group of four rows keeps
/// the union of their nonzero columns k and the values interleaved, `values[(i * 4) + row % 4]` for
/// the i-th column of the group.
#[derive(Default)]
struct PackedRows {
    groups: Vec<(Vec<usize>, Vec<f32>)>,
    out_rows: Vec<usize>, // output row of each packed row (usize::MAX for padding)
}

impl PackedRows {
    /// Pack the rows in the given order; row `r` has the nonzero (k, value) pairs `row(r)`.
    fn pack(&mut self, order: &[usize], row: impl Fn(usize) -> Vec<(usize, f32)>) {
        self.groups.clear();
        self.out_rows.clear();
        for chunk in order.chunks(4) {
            let rows: Vec<Vec<(usize, f32)>> = chunk.iter().map(|&r| row(r)).collect();
            let mut ks: Vec<usize> = rows.iter().flatten().map(|&(k, _)| k).collect();
            ks.sort_unstable();
            ks.dedup();
            let mut values = vec![0.0f32; ks.len() * 4];
            for (g, entries) in rows.iter().enumerate() {
                for &(k, v) in entries {
                    let i = ks.binary_search(&k).unwrap();
                    values[i * 4 + g] = v;
                }
            }
            self.groups.push((ks, values));
            self.out_rows.extend(chunk);
            self.out_rows.extend(std::iter::repeat_n(usize::MAX, 4 - chunk.len()));
        }
    }
}

/// Columns of a `gemm` register block: 4 x STRIP accumulators, 12 AVX registers, enough independent
/// FMA chains to keep both FMA units busy.
const STRIP: usize = 24;

/// out[out_row * ldo + j] = sum_k A[row][k] * b[k * ldb + j] for every packed row of A and j < cols
/// (a multiple of STRIP). Blocks of 4 rows x STRIP columns stay in registers over the group's columns k;
/// the columns go outside so that the strips of b stay in L1 for all the groups.
fn gemm(a: &PackedRows, b: &[f32], ldb: usize, cols: usize, out: &mut [f32], ldo: usize) {
    for j0 in (0..cols).step_by(STRIP) {
        for (gi, (ks, values)) in a.groups.iter().enumerate() {
            let out_rows = &a.out_rows[gi * 4..gi * 4 + 4];
            let mut acc = [[0.0f32; STRIP]; 4];
            for (i, &k) in ks.iter().enumerate() {
                let brow: &[f32; STRIP] = b[k * ldb + j0..k * ldb + j0 + STRIP].try_into().unwrap();
                let av: &[f32; 4] = values[i * 4..i * 4 + 4].try_into().unwrap();
                for g in 0..4 {
                    for l in 0..STRIP {
                        acc[g][l] = av[g].mul_add(brow[l], acc[g][l]);
                    }
                }
            }
            for g in 0..4 {
                if out_rows[g] != usize::MAX {
                    let start = out_rows[g] * ldo + j0;
                    out[start..start + STRIP].copy_from_slice(&acc[g]);
                }
            }
        }
    }
}

fn round_up(x: usize, to: usize) -> usize {
    x.div_ceil(to) * to
}

fn sum32(x: &[f32]) -> f64 {
    let mut acc = [0.0f32; 16];
    let xc = x.chunks_exact(16);
    let xr = xc.remainder();
    for a in xc {
        for l in 0..16 {
            acc[l] += a[l];
        }
    }
    acc.iter().map(|&v| v as f64).sum::<f64>() + xr.iter().map(|&v| v as f64).sum::<f64>()
}

fn matvec(w: &[f64], x: &[f64]) -> Vec<f64> {
    (0..N).map(|a| dot(&w[a * N..(a + 1) * N], x)).collect()
}

impl<'a> Game<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: usize,
        w: &'a [f64],
        eq: &'a [f64],
        tie: &'a [f64],
        cc: &'a [f64],
        v_steal: &[f64],
        v_iwin: &[f64],
        v_jwin: &[f64],
        v_tie: &[f64],
        v3: &[f64],
        mw: Option<Multiway<'a>>,
        dead: Option<Dead<'a>>,
    ) -> Self {
        let mut pairs = Vec::new();
        for i in 0..m - 1 {
            for j in i + 1..m {
                pairs.push((i, j));
            }
        }
        let mut triples = Vec::new();
        let mut triple_pair = Vec::new();
        let mut pair_triples = vec![Vec::new(); pairs.len()];
        if mw.is_some() {
            for i in 0..m.saturating_sub(2) {
                for j in i + 1..m - 1 {
                    for k in j + 1..m {
                        let p = pairs.iter().position(|&x| x == (i, j)).unwrap();
                        pair_triples[p].push(triples.len());
                        triple_pair.push(p);
                        triples.push((i, j, k));
                    }
                }
            }
        }
        let rows = |data: &[f64], len: usize, count: usize| -> Vec<Vec<f64>> {
            (0..count).map(|r| data[r * len..(r + 1) * len].to_vec()).collect()
        };
        let v3 = rows(v3, OUTCOMES * m, triples.len());
        let wsum = (0..N).map(|a| w[a * N..(a + 1) * N].iter().sum()).collect();
        Game {
            m,
            w,
            eq,
            tie,
            wsum,
            w32: w.iter().map(|&x| x as f32).collect(),
            dead_by_rank: dead.as_ref().map(|d| {
                let by_rank = |t: &[f64]| -> Vec<f64> { (0..RANKS * N * N).map(|i| t[(i % (N * N)) * RANKS + i / (N * N)]).collect() };
                (by_rank(d.de), by_rank(d.dt))
            }),
            dead,
            cc,
            v_steal: rows(v_steal, m, m),
            v_iwin: rows(v_iwin, m, pairs.len()),
            v_jwin: rows(v_jwin, m, pairs.len()),
            v_tie: rows(v_tie, m, pairs.len()),
            alpha: v3
                .iter()
                .map(|r| {
                    let g = |pair: usize, x: usize| (0..ORDERS).map(|o| PAIR_SHIFT[o][pair] * r[o * m + x]).sum::<f64>();
                    let roles = [[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, -1.0]];
                    roles
                        .iter()
                        .flat_map(|c| (0..m).map(move |x| (c[0] * g(0, x) + c[1] * g(1, x) + c[2] * g(2, x)) as f32))
                        .collect()
                })
                .collect(),
            v3: v3.iter().map(|r| r.iter().map(|&v| v as f32).collect()).collect(),
            pairs,
            triples,
            pair_triples,
            triple_pair,
            mw,
            minor: minor_layouts(),
        }
    }

    fn pair_index(&self, i: usize, j: usize) -> usize {
        // pairs are ordered (0,1), (0,2), .., (0,m-1), (1,2), ..
        i * (2 * self.m - i - 1) / 2 + (j - i - 1)
    }

    pub fn initial(&self) -> Strategies {
        Strategies {
            push: vec![vec![0.5; N]; self.m - 1],
            call: vec![vec![0.5; N]; self.pairs.len()],
            over: vec![vec![0.5; N]; self.triples.len()],
        }
    }

    /// For a player folding the hands in `fold`: the expected number of cards of each rank he holds when
    /// he folds, and phi[x] = P(he folds | another player holds x) / P(he folds).
    fn fold_stats(&self, fold: &[f64]) -> FoldStats {
        let mass = dot(fold, self.cc);
        match &self.dead {
            Some(d) if mass > 1e-12 => {
                let mut nu = vec![0.0; RANKS];
                for x in 0..N {
                    let w = fold[x] * self.cc[x];
                    for r in 0..RANKS {
                        nu[r] += w * d.ranks[x * RANKS + r];
                    }
                }
                nu.iter_mut().for_each(|v| *v /= mass);
                let wf = matvec(self.w, fold);
                let phi = (0..N).map(|x| wf[x] / self.wsum[x] / (mass / COMBOS)).collect();
                (nu, phi)
            }
            _ => (vec![0.0; RANKS], vec![1.0; N]),
        }
    }

    /// E (or T with `tie`) moved by the dead cards nu; None if the cached one in `slot` was built for
    /// about the same nu.
    fn moved(&self, tie: bool, nu: &[f64], slot: Option<&(Vec<f64>, impl Sized)>) -> Option<Vec<f64>> {
        let (de, dt) = self.dead_by_rank.as_ref()?;
        if slot.is_some_and(|(old, _)| old.iter().zip(nu).all(|(a, b)| (a - b).abs() <= DEAD_TOL)) {
            return None;
        }
        let table = if tie { dt } else { de };
        let mut out = if tie { self.tie } else { self.eq }.to_vec();
        for (r, &c) in nu.iter().enumerate() {
            if c != 0.0 {
                out.iter_mut().zip(&table[r * NN..(r + 1) * NN]).for_each(|(o, d)| *o = c.mul_add(*d, *o));
            }
        }
        Some(out)
    }

    /// Equities and ties of pair p given the folds before the caller.
    fn pair_equities<'c>(&'c self, p: usize, cache: &'c DeadCache) -> (&'c [f64], &'c [f64]) {
        match (cache.slots.get(2 * p), cache.slots.get(2 * p + 1)) {
            (Some(Some((_, e))), Some(Some((_, t)))) => (e, t),
            _ => (self.eq, self.tie),
        }
    }

    /// The pair sums with the caller's range c, the shover's range pe and the pair's phi; without
    /// equities only wsum, wc and wp.
    fn pair_sums(&self, c: &[f64], pe: &[f64], phi: &[f64], equities: Option<(&[f64], &[f64])>) -> PairSums {
        let mut ps = PairSums {
            wsum: vec![0.0; N],
            wc: vec![0.0; N],
            wec: vec![0.0; N],
            wtc: vec![0.0; N],
            wp: vec![0.0; N],
            wep: vec![0.0; N],
            wtp: vec![0.0; N],
        };
        let mut w = [0.0f64; N];
        for a in 0..N {
            let row = &self.w[a * N..(a + 1) * N];
            for b in 0..N {
                w[b] = row[b] * phi[a] * phi[b];
            }
            ps.wsum[a] = w.iter().sum();
            ps.wc[a] = dot(&w, c);
            let x = pe[a];
            ps.wp.iter_mut().zip(&w).for_each(|(d, v)| *d = v.mul_add(x, *d));
            if let Some((e, t)) = equities {
                let (er, tr) = (&e[a * N..(a + 1) * N], &t[a * N..(a + 1) * N]);
                let (mut sec, mut stc) = (0.0, 0.0);
                for b in 0..N {
                    let wb = w[b];
                    sec = (wb * er[b]).mul_add(c[b], sec);
                    stc = (wb * tr[b]).mul_add(c[b], stc);
                    ps.wep[b] = (wb * x).mul_add(e[b * N + a], ps.wep[b]);
                    ps.wtp[b] = (wb * x).mul_add(tr[b], ps.wtp[b]);
                }
                ps.wec[a] = sec;
                ps.wtc[a] = stc;
            }
        }
        ps
    }

    fn bunching(&self, s: &Strategies, multiway: bool, cache: &mut DeadCache) -> Bunch {
        let m = self.m;
        let np = self.pairs.len();
        let nt = if multiway { self.triples.len() } else { 0 };
        let folds = |x: &Vec<f64>| -> Vec<f64> { x.iter().map(|v| 1.0 - v).collect() };
        let push_st: Vec<FoldStats> = s.push.iter().map(|x| self.fold_stats(&folds(x))).collect();
        let call_st: Vec<FoldStats> = s.call.iter().map(|x| self.fold_stats(&folds(x))).collect();
        let over_st: Vec<FoldStats> = s.over[..nt].iter().map(|x| self.fold_stats(&folds(x))).collect();
        let combine = |stats: &[&FoldStats]| -> FoldStats {
            let (mut nu, mut phi) = (vec![0.0; RANKS], vec![1.0; N]);
            for (n, f) in stats {
                nu.iter_mut().zip(n).for_each(|(a, b)| *a += b);
                phi.iter_mut().zip(f).for_each(|(a, b)| *a *= b);
            }
            (nu, phi)
        };
        cache.slots.resize(2 * np, None);
        cache.show.resize(2 * np, None);
        let mut b = Bunch {
            pair_phi: Vec::with_capacity(np),
            pair_wsum: Vec::with_capacity(np),
            over_phi: vec![Vec::new(); nt],
            strength: vec![Vec::new(); nt],
            opener: (0..m).map(|i| combine(&push_st[..i].iter().collect::<Vec<_>>()).1).collect(),
        };
        for (p, &(i, j)) in self.pairs.iter().enumerate() {
            let mut before: Vec<&FoldStats> = push_st[..i].iter().collect();
            before.extend((i + 1..j).map(|x| &call_st[self.pair_index(i, x)]));
            let (nu, phi) = combine(&before);
            for (k, tie) in [(2 * p, false), (2 * p + 1, true)] {
                if let Some(out) = self.moved(tie, &nu, cache.slots[k].as_ref()) {
                    cache.slots[k] = Some((nu.clone(), out));
                }
            }
            let wphi = matvec(self.w, &phi);
            b.pair_wsum.push((0..N).map(|a| phi[a] * wphi[a]).collect());
            if multiway {
                let tri = &self.pair_triples[p];
                let all: Vec<&FoldStats> = tri.iter().map(|&t| &over_st[t]).collect();
                let nu_all: Vec<f64> = nu.iter().zip(&combine(&all).0).map(|(a, c)| a + c).collect();
                for (k, tie) in [(2 * p, false), (2 * p + 1, true)] {
                    let base = if tie { self.tie } else { self.eq };
                    match self.moved(tie, &nu_all, cache.show[k].as_ref()) {
                        Some(out) => cache.show[k] = Some((nu_all.clone(), out.iter().map(|&v| v as f32).collect())),
                        None if cache.show[k].is_none() => {
                            cache.show[k] = Some((nu_all.clone(), base.iter().map(|&v| v as f32).collect()))
                        }
                        None => {}
                    }
                }
                for (n, &t) in tri.iter().enumerate() {
                    let (nu_k, phi_k) = combine(&all[..n]);
                    b.over_phi[t] = phi.iter().zip(&phi_k).map(|(a, c)| a * c).collect();
                    b.strength[t] = match &self.dead {
                        Some(d) => (0..N)
                            .map(|c| (0..RANKS).map(|r| d.strength[c * RANKS + r] * (nu[r] + nu_k[r])).sum())
                            .collect(),
                        None => vec![0.0; N],
                    };
                }
            }
            b.pair_phi.push(phi);
        }
        b
    }

    /// Call probability at the first-caller node (i, j), over i's range.
    fn call_prob(&self, s: &Strategies, p: usize, wc: &[f64], wsum: &[f64]) -> f64 {
        let pe: Vec<f64> = s.push[self.pairs[p].0].iter().map(|x| x + 1e-12).collect();
        dot(&pe, wc) / dot(&pe, wsum)
    }

    /// Call probability, shover equity and tie probability at the first-caller node (i, j), over i's range.
    fn caller_stats(&self, s: &Strategies, p: usize, wc: &[f64], wec: &[f64], wtc: &[f64], wsum: &[f64]) -> (f64, f64, f64) {
        let pe: Vec<f64> = s.push[self.pairs[p].0].iter().map(|x| x + 1e-12).collect();
        let num = dot(&pe, wc);
        let pwin = if num > 1e-12 { dot(&pe, wec) / num } else { 0.5 };
        let ptie = if num > 1e-12 { dot(&pe, wtc) / num } else { 0.0 };
        (self.call_prob(s, p, wc, wsum), pwin, ptie)
    }

    /// Value vector after i shoves and the players before `from` have folded.
    fn after_shove(&self, i: usize, from: usize, cps: &[f64], called: &[Vec<f64>]) -> Vec<f64> {
        let mut acc = vec![0.0; self.m];
        let mut q = 1.0;
        for k in from..self.m {
            let p = self.pair_index(i, k);
            for x in 0..self.m {
                acc[x] += q * cps[p] * called[p][x];
            }
            q *= 1.0 - cps[p];
        }
        for x in 0..self.m {
            acc[x] += q * self.v_steal[i][x];
        }
        acc
    }

    /// node[i] = value vector when it is folded to player i.
    fn node_values(&self, s: &Strategies, cps: &[f64], called: &[Vec<f64>], opener: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = self.m;
        let mut node = vec![Vec::new(); m];
        node[m - 1] = self.v_steal[m - 1].clone();
        for i in (0..m - 1).rev() {
            let w: Vec<f64> = self.cc.iter().zip(&opener[i]).map(|(c, f)| c * f).collect();
            let ppush = dot(&s.push[i], &w) / w.iter().sum::<f64>();
            let after = self.after_shove(i, i + 1, cps, called);
            node[i] = (0..m).map(|x| ppush * after[x] + (1.0 - ppush) * node[i + 1][x]).collect();
        }
        node
    }

    /// EVs against the strategies `s`; `cache` keeps the dead-card equities between calls.
    pub fn evaluate(&self, s: &Strategies, multiway: bool, threads: usize, mw_min_freq: f64, cache: &mut DeadCache) -> Ev {
        if multiway {
            self.evaluate_multiway(s, threads, mw_min_freq, cache)
        } else {
            self.evaluate_hu(s, cache)
        }
    }

    fn evaluate_hu(&self, s: &Strategies, cache: &mut DeadCache) -> Ev {
        let m = self.m;
        let bn = self.bunching(s, false, cache);
        let sums: Vec<PairSums> = self
            .pairs
            .iter()
            .enumerate()
            .map(|(p, &(i, _))| {
                let pe: Vec<f64> = s.push[i].iter().map(|x| x + 1e-12).collect();
                self.pair_sums(&s.call[p], &pe, &bn.pair_phi[p], Some(self.pair_equities(p, cache)))
            })
            .collect();
        let mut cps = vec![0.0; self.pairs.len()];
        let mut called = vec![Vec::new(); self.pairs.len()];
        for (p, ps) in sums.iter().enumerate() {
            let (cp, pwin, ptie) = self.caller_stats(s, p, &ps.wc, &ps.wec, &ps.wtc, &ps.wsum);
            cps[p] = cp;
            called[p] = (0..m)
                .map(|x| pwin * self.v_iwin[p][x] + (1.0 - pwin) * self.v_jwin[p][x] + ptie * self.v_tie[p][x])
                .collect();
        }
        let node = self.node_values(s, &cps, &called, &bn.opener);

        let mut ev = Ev { push: Vec::new(), call: vec![(Vec::new(), 0.0); self.pairs.len()], over: Vec::new() };
        for i in 0..m - 1 {
            let mut ev_push = vec![0.0; N];
            let mut q = vec![1.0; N];
            for j in i + 1..m {
                let ps = &sums[self.pair_index(i, j)];
                let p = self.pair_index(i, j);
                for a in 0..N {
                    let callp = ps.wc[a] / ps.wsum[a];
                    let eqc = if ps.wc[a] > 1e-12 { ps.wec[a] / ps.wc[a] } else { 0.5 };
                    let tiec = if ps.wc[a] > 1e-12 { ps.wtc[a] / ps.wc[a] } else { 0.0 };
                    ev_push[a] += q[a]
                        * callp
                        * (eqc * self.v_iwin[p][i] + (1.0 - eqc) * self.v_jwin[p][i] + tiec * self.v_tie[p][i]);
                    q[a] *= 1.0 - callp;
                }
            }
            for a in 0..N {
                ev_push[a] += q[a] * self.v_steal[i][i];
            }
            ev.push.push((ev_push, node[i + 1][i]));

            for j in i + 1..m {
                let p = self.pair_index(i, j);
                let ps = &sums[p];
                let ev_call = (0..N)
                    .map(|b| {
                        let (eqp, tiep) = (ps.wep[b] / ps.wp[b], ps.wtp[b] / ps.wp[b]);
                        eqp * self.v_jwin[p][j] + (1.0 - eqp) * self.v_iwin[p][j] + tiep * self.v_tie[p][j]
                    })
                    .collect();
                let fold = self.after_shove(i, j + 1, &cps, &called)[j];
                ev.call[p] = (ev_call, fold);
            }
        }
        ev
    }

    fn evaluate_multiway(&self, s: &Strategies, threads: usize, min_freq: f64, cache: &mut DeadCache) -> Ev {
        let t_setup = Instant::now();
        let m = self.m;
        let np = self.pairs.len();
        let nt = self.triples.len();
        let bn = self.bunching(s, true, cache);
        let f32v = |x: &Vec<f64>| -> Vec<f32> { x.iter().map(|&v| v as f32).collect() };
        let prune = |x: &Vec<f64>| -> Vec<f32> {
            x.iter().map(|&v| if v < min_freq { 0.0 } else { v as f32 }).collect()
        };
        let push32: Vec<Vec<f32>> = s.push.iter().map(f32v).collect();
        let call32: Vec<Vec<f32>> = s.call.iter().map(f32v).collect();
        let push_eps: Vec<Vec<f32>> = push32.iter().map(|x| x.iter().map(|v| v + 1e-12).collect()).collect();
        let push_pr: Vec<Vec<f32>> = s.push.iter().map(prune).collect();
        let call_pr: Vec<Vec<f32>> = s.call.iter().map(prune).collect();
        // the overcalling ranges with the overcaller's phi
        let over_pr: Vec<Vec<f32>> = s
            .over
            .iter()
            .zip(&bn.over_phi)
            .map(|(x, f)| prune(x).iter().zip(f).map(|(v, f)| v * *f as f32).collect())
            .collect();
        let call_support: Vec<Vec<usize>> =
            call_pr.iter().map(|c| (0..N).filter(|&b| c[b] != 0.0).collect()).collect();
        // the hands in some overcalling range, and the ranges packed for `gemm` with similar ranges
        // grouped together (by size), over the positions of their hands in `c_union`
        let c_union: Vec<usize> = (0..N).filter(|&c| over_pr.iter().any(|v| v[c] != 0.0)).collect();
        let mut c_pos = vec![0usize; N];
        for (ci, &c) in c_union.iter().enumerate() {
            c_pos[c] = ci;
        }
        let mut order: Vec<usize> = (0..nt).collect();
        let support_len = |t: usize| over_pr[t].iter().filter(|&&v| v != 0.0).count();
        order.sort_by_key(|&t| (std::cmp::Reverse(support_len(t)), t));
        let mut over_packed = PackedRows::default();
        over_packed.pack(&order, |t| {
            (0..N).filter(|&c| over_pr[t][c] != 0.0).map(|c| (c_pos[c], over_pr[t][c])).collect()
        });
        let mut over_strength_packed = PackedRows::default();
        over_strength_packed.pack(&order, |t| {
            (0..N)
                .filter(|&c| over_pr[t][c] != 0.0)
                .map(|c| (c_pos[c], over_pr[t][c] * bn.strength[t][c] as f32))
                .collect()
        });
        let pair_phi32: Vec<Vec<f32>> = bn.pair_phi.iter().map(f32v).collect();
        let strength32: Vec<Vec<f32>> = bn.strength.iter().map(f32v).collect();

        // q[p][a]: probability that the players after the shover and before the caller of pair p fold,
        // and the same past all players for the last entry of each shover
        let sums: Vec<PairSums> = self
            .pairs
            .iter()
            .enumerate()
            .map(|(p, &(i, _))| {
                let pe: Vec<f64> = s.push[i].iter().map(|x| x + 1e-12).collect();
                self.pair_sums(&s.call[p], &pe, &bn.pair_phi[p], None)
            })
            .collect();
        let callmass: Vec<&Vec<f64>> = sums.iter().map(|ps| &ps.wc).collect();
        let mut q_before = vec![Vec::new(); np];
        let mut q_all = vec![Vec::new(); m - 1];
        for i in 0..m - 1 {
            let mut q = vec![1.0f64; N];
            for j in i + 1..m {
                let p = self.pair_index(i, j);
                q_before[p] = q.clone();
                for a in 0..N {
                    q[a] *= 1.0 - callmass[p][a] / bn.pair_wsum[p][a];
                }
            }
            q_all[i] = q;
        }
        let push_den: Vec<&Vec<f64>> = sums.iter().map(|ps| &ps.wp).collect();

        let show: Vec<(&[f32], &[f32])> = (0..np)
            .map(|p| match (&cache.show[2 * p], &cache.show[2 * p + 1]) {
                (Some((_, e)), Some((_, t))) => (&e[..], &t[..]),
                _ => unreachable!("the showdown equities are built with the overcalls"),
            })
            .collect();
        let ctx = RowContext {
            g: self,
            push32: &push32,
            call32: &call32,
            push_eps: &push_eps,
            push_pr: &push_pr,
            call_pr: &call_pr,
            call_support: &call_support,
            c_union: &c_union,
            over_packed: &over_packed,
            over_strength_packed: &over_strength_packed,
            q_before: &q_before,
            bunch: &bn,
            show: &show,
            pair_phi: &pair_phi32,
            strength: &strength32,
        };
        let setup_ms = t_setup.elapsed().as_secs_f64() * 1e3;
        let threads = threads.clamp(1, N.div_ceil(ROW_BLOCK));
        let acc = thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|tid| {
                    let ctx = &ctx;
                    scope.spawn(move || {
                        let mut acc = Accumulators::new(m, np, nt);
                        let mut ws = Workspace::new(m, nt);
                        let mut block = tid;
                        while block * ROW_BLOCK < N {
                            let a0 = block * ROW_BLOCK;
                            ctx.block(a0, (a0 + ROW_BLOCK).min(N), &mut ws, &mut acc);
                            block += threads;
                        }
                        acc
                    })
                })
                .collect();
            let mut total = Accumulators::new(m, np, nt);
            for h in handles {
                total.add(&h.join().unwrap());
            }
            total
        });

        if std::env::var_os("MBR_PROFILE").is_some() {
            eprintln!(
                "multiway eval: setup {:.1} ms, rows (thread-summed) pass A {:.1} ms, pairs {:.1} ms, pass B {:.1} ms",
                setup_ms, acc.timing[0] * 1e3, acc.timing[1] * 1e3, acc.timing[2] * 1e3
            );
        }
        let cps: Vec<f64> = (0..np).map(|p| self.call_prob(s, p, callmass[p], &bn.pair_wsum[p])).collect();
        let called: Vec<Vec<f64>> = (0..np)
            .map(|p| acc.called[p].iter().map(|v| v / acc.called_w[p].max(1e-12)).collect())
            .collect();
        let node = self.node_values(s, &cps, &called, &bn.opener);

        let mut ev = Ev { push: Vec::new(), call: vec![(Vec::new(), 0.0); np], over: Vec::new() };
        for i in 0..m - 1 {
            let ev_push = (0..N).map(|a| acc.push[i][a] + q_all[i][a] * self.v_steal[i][i]).collect();
            ev.push.push((ev_push, node[i + 1][i]));
            for j in i + 1..m {
                let p = self.pair_index(i, j);
                let ev_call = (0..N).map(|b| acc.call[p][b] / push_den[p][b]).collect();
                ev.call[p] = (ev_call, self.after_shove(i, j + 1, &cps, &called)[j]);
            }
        }
        for t in 0..nt {
            let k = self.triples[t].2;
            let fold = acc.fold_num[t] / acc.fold_den[t].max(1e-12);
            let al_c = self.alpha[t][2 * m + k] as f64;
            let ev_over = (0..N)
                .map(|c| {
                    let x = |o: usize| acc.x[(t * SLOTS + o) * N + c];
                    let den: f64 = (0..ORDERS).map(x).sum();
                    let num: f64 = (0..OUTCOMES).map(|o| self.v3[t][o * m + k] as f64 * x(o)).sum::<f64>()
                        + x(OUTCOMES)
                        + al_c * bn.strength[t][c] * den;
                    if den > 1e-9 { num / den } else { fold - 1.0 }
                })
                .collect();
            ev.over.push((ev_over, fold));
        }
        ev
    }
}

/// Per-thread sums over the grid rows, added up at the end.
struct Accumulators {
    push: Vec<Vec<f64>>,  // [i][a] sum over b of the shover's value, times q_before
    call: Vec<Vec<f64>>,  // [p][b] sum over a of the caller's value weighted by the shover's range
    called: Vec<Vec<f64>>, // [p][x] value vectors once the caller of p has called, weighted
    called_w: Vec<f64>,
    fold_num: Vec<f64>, // [t] the overcaller's value when folding, weighted
    fold_den: Vec<f64>,
    x: Vec<f64>, // [t][slot][c] sum over (a, b) of the line weight times U_o[a, b, c], then the shifted line weight times W3
    timing: [f64; 3], // seconds in pass A, the pairs and pass B, for MBR_PROFILE
}

impl Accumulators {
    fn new(m: usize, np: usize, nt: usize) -> Self {
        Accumulators {
            push: vec![vec![0.0; N]; m - 1],
            call: vec![vec![0.0; N]; np],
            called: vec![vec![0.0; m]; np],
            called_w: vec![0.0; np],
            fold_num: vec![0.0; nt],
            fold_den: vec![0.0; nt],
            x: vec![0.0; nt * SLOTS * N],
            timing: [0.0; 3],
        }
    }

    fn add(&mut self, o: &Accumulators) {
        let add = |a: &mut Vec<f64>, b: &Vec<f64>| a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        self.push.iter_mut().zip(&o.push).for_each(|(a, b)| add(a, b));
        self.call.iter_mut().zip(&o.call).for_each(|(a, b)| add(a, b));
        self.called.iter_mut().zip(&o.called).for_each(|(a, b)| add(a, b));
        add(&mut self.called_w, &o.called_w);
        add(&mut self.fold_num, &o.fold_num);
        add(&mut self.fold_den, &o.fold_den);
        add(&mut self.x, &o.x);
        for k in 0..3 {
            self.timing[k] += o.timing[k];
        }
    }
}

/// Per-thread buffers for a block of grid rows.
struct Workspace {
    s: Vec<f32>,    // [t][slot][row in block * N + b] (stride BLOCK_COLS) sum over c of U_o[a, b, c] * over_t[c],
                    // then of W3[a, b, c] * over_t[c] * strength_t[c]
    pack: Vec<f32>, // pass A: [overcaller hand][row in block * N + b] of one order tensor
    pack_b: Vec<f32>, // pass B: [caller hand][c] (stride PAD_N)
    mat_packed: PackedRows, // pass B: line weights [active triple][caller hand], packed for gemm
    shifted_packed: PackedRows, // the line weights times the shover's and caller's dead-card strength
    x_row: Vec<f32>, // pass B: [active triple][c] (stride PAD_N)
    b_union: Vec<usize>,
    in_union: Vec<bool>,
    cur: Vec<f32>,  // [x][b] value of player x in the current pair's backward recursion
    tmp: Vec<f32>,  // [b]
    over_mass: Vec<f32>, // [b] sum over c of W3[a, b, c] * over_t[c]
    op: Vec<f32>,   // [k][b] overcall probability of player k
    reach: Vec<f32>, // [b]
    wc: Vec<f32>,   // [b] W[a, b] * call[b]
    mat: Vec<f32>,  // [t][b] line weight of triple t for pass B
    active: Vec<usize>, // triples with a line weight in this row
}

impl Workspace {
    fn new(m: usize, nt: usize) -> Self {
        Workspace {
            s: vec![0.0; round_up(nt, 4) * SLOTS * BLOCK_COLS],
            pack: vec![0.0; N * BLOCK_COLS],
            pack_b: vec![0.0; N * PAD_N],
            mat_packed: PackedRows::default(),
            shifted_packed: PackedRows::default(),
            x_row: vec![0.0; round_up(nt, 4) * PAD_N],
            b_union: Vec::with_capacity(N),
            in_union: vec![false; N],
            cur: vec![0.0; m * N],
            tmp: vec![0.0; N],
            over_mass: vec![0.0; N],
            op: vec![0.0; m * N],
            reach: vec![0.0; N],
            wc: vec![0.0; N],
            mat: vec![0.0; nt * N],
            active: Vec::with_capacity(nt),
        }
    }
}

struct RowContext<'g, 'a> {
    g: &'g Game<'a>,
    push32: &'g [Vec<f32>],
    call32: &'g [Vec<f32>],
    push_eps: &'g [Vec<f32>],
    push_pr: &'g [Vec<f32>],
    call_pr: &'g [Vec<f32>],
    call_support: &'g [Vec<usize>],
    c_union: &'g [usize],
    over_packed: &'g PackedRows,
    over_strength_packed: &'g PackedRows,
    q_before: &'g [Vec<f64>],
    bunch: &'g Bunch,
    show: &'g [(&'g [f32], &'g [f32])],
    pair_phi: &'g [Vec<f32>],
    strength: &'g [Vec<f32>],
}

impl RowContext<'_, '_> {
    /// Grid rows a0 .. a1: pass A for all of them at once, so that the tensors are read in long runs,
    /// then the rest row by row.
    fn block(&self, a0: usize, a1: usize, ws: &mut Workspace, acc: &mut Accumulators) {
        let mw = self.g.mw.as_ref().unwrap();
        let t_start = Instant::now();
        // pass A: S[t][o][a][b] = sum_c over_t[c] U_o[a, b, c], a matrix product per order over the
        // overcalling hands, with the tensor rows of this block copied next to each other first
        let len = (a1 - a0) * N;
        for o in 0..SLOTS {
            let (tensor, rows) = if o < OUTCOMES { (mw.t[o], self.over_packed) } else { (mw.mass, self.over_strength_packed) };
            for (ci, &c) in self.c_union.iter().enumerate() {
                ws.pack[ci * BLOCK_COLS..ci * BLOCK_COLS + len].copy_from_slice(&tensor[c * NN + a0 * N..c * NN + a1 * N]);
            }
            gemm(rows, &ws.pack, BLOCK_COLS, BLOCK_COLS, &mut ws.s[o * BLOCK_COLS..], SLOTS * BLOCK_COLS);
        }
        acc.timing[0] += t_start.elapsed().as_secs_f64();
        for a in a0..a1 {
            self.row(a, a - a0, ws, acc);
        }
    }

    /// Everything else the multiway evaluation needs from grid row a (shover hand a), the r-th row of
    /// its block.
    fn row(&self, a: usize, r: usize, ws: &mut Workspace, acc: &mut Accumulators) {
        let g = self.g;
        let mw = g.mw.as_ref().unwrap();
        let m = g.m;
        let r0 = a * N;
        let wrow = &g.w32[r0..r0 + N];
        let inv_d: Vec<f32> = mw.d_all[r0..r0 + N].iter().map(|d| 1.0 / d).collect();
        let sidx = |t: usize, o: usize| (t * SLOTS + o) * BLOCK_COLS + r * N;

        let t_a = Instant::now();
        ws.active.clear();
        for (p, &(i, j)) in g.pairs.iter().enumerate() {
            let tri = &g.pair_triples[p];
            // overcall probabilities of the players after the caller
            for (n, &t) in tri.iter().enumerate() {
                let k = j + 1 + n;
                let op = &mut ws.op[k * N..(k + 1) * N];
                op.copy_from_slice(&ws.s[sidx(t, 0)..sidx(t, 0) + N]);
                for o in 1..ORDERS {
                    let so = &ws.s[sidx(t, o)..sidx(t, o) + N];
                    op.iter_mut().zip(so).for_each(|(d, x)| *d += x);
                }
                // at most 1: with a full overcalling range rounding could push it over and make the
                // weights of the lines where k folds negative
                op.iter_mut().zip(&inv_d).for_each(|(d, x)| *d = (*d * x).min(1.0));
            }
            // line weights for the overcallers: shover, caller, and the players in between folding
            let phi = &self.pair_phi[p];
            let pa = self.push_pr[i][a];
            let line = pa != 0.0 && !self.call_support[p].is_empty();
            if line {
                ws.reach.iter_mut().for_each(|r| *r = pa * phi[a]);
                for (n, &t) in tri.iter().enumerate() {
                    let k = j + 1 + n;
                    let mat = &mut ws.mat[t * N..(t + 1) * N];
                    for b in 0..N {
                        mat[b] = ws.reach[b] * self.call_pr[p][b] * phi[b];
                    }
                    let op = &ws.op[k * N..(k + 1) * N];
                    ws.reach.iter_mut().zip(op).for_each(|(r, o)| *r *= 1.0 - o);
                    ws.active.push(t);
                }
            }
            // backward recursion from the heads-up showdown to the player after the caller
            let (vi, vj, vt) = (&g.v_iwin[p], &g.v_jwin[p], &g.v_tie[p]);
            let erow = &self.show[p].0[r0..r0 + N];
            let trow = &self.show[p].1[r0..r0 + N];
            for x in 0..m {
                let (wi, wj, wt) = (vi[x] as f32, vj[x] as f32, vt[x] as f32);
                let cur = &mut ws.cur[x * N..(x + 1) * N];
                for b in 0..N {
                    cur[b] = wi * erow[b] + wj * (1.0 - erow[b]) + wt * trow[b];
                }
            }
            for (n, &t) in tri.iter().enumerate().rev() {
                let k = j + 1 + n;
                if line {
                    // the overcaller k folding: everybody behind k as in the recursion so far
                    let mat = &ws.mat[t * N..(t + 1) * N];
                    for b in 0..N {
                        ws.tmp[b] = wrow[b] * mat[b];
                    }
                    acc.fold_num[t] += dot32(&ws.tmp, &ws.cur[k * N..(k + 1) * N]);
                    acc.fold_den[t] += sum32(&ws.tmp);
                }
                // cur = (1 - op_k) cur + sum_o v3[o] S_o / D, reading each S_o once
                let v3 = &g.v3[t];
                let op = &ws.op[k * N..(k + 1) * N];
                for x in 0..m {
                    let cur = &mut ws.cur[x * N..(x + 1) * N];
                    cur.iter_mut().zip(op).for_each(|(c, o)| *c *= 1.0 - o);
                }
                for o in 0..OUTCOMES {
                    let so = &ws.s[sidx(t, o)..sidx(t, o) + N];
                    ws.tmp.iter_mut().zip(so).zip(&inv_d).for_each(|((d, s), i)| *d = s * i);
                    for x in 0..m {
                        let coef = v3[o * m + x];
                        let cur = &mut ws.cur[x * N..(x + 1) * N];
                        cur.iter_mut().zip(&ws.tmp).for_each(|(c, s)| *c = coef.mul_add(*s, *c));
                    }
                }
                // the dead cards: (alpha_0 st[a] + alpha_1 st[b]) sum_c W3 over + alpha_2 sum_c W3 over st
                ws.over_mass.copy_from_slice(&ws.s[sidx(t, 0)..sidx(t, 0) + N]);
                for o in 1..ORDERS {
                    let so = &ws.s[sidx(t, o)..sidx(t, o) + N];
                    ws.over_mass.iter_mut().zip(so).for_each(|(d, x)| *d += x);
                }
                let (st, al) = (&self.strength[t], &g.alpha[t]);
                let shifted = &ws.s[sidx(t, OUTCOMES)..sidx(t, OUTCOMES) + N];
                for x in 0..m {
                    let (a0, a1, a2) = (al[x] * st[a], al[m + x], al[2 * m + x]);
                    let cur = &mut ws.cur[x * N..(x + 1) * N];
                    for b in 0..N {
                        cur[b] += ((a0 + a1 * st[b]) * ws.over_mass[b] + a2 * shifted[b]) * inv_d[b];
                    }
                }
            }
            // shover's and caller's values, and the value vectors once j has called
            // with the pair weights W[a, b] phi[a] phi[b]
            let c32 = &self.call32[p];
            for b in 0..N {
                ws.wc[b] = wrow[b] * phi[a] * phi[b] * c32[b];
            }
            acc.push[i][a] += self.q_before[p][a] * dot32(&ws.wc, &ws.cur[i * N..(i + 1) * N]) / self.bunch.pair_wsum[p][a];
            let pe = self.push_eps[i][a];
            let curj = &ws.cur[j * N..(j + 1) * N];
            for (b, dst) in acc.call[p].iter_mut().enumerate() {
                *dst += (wrow[b] * phi[a] * phi[b] * pe * curj[b]) as f64;
            }
            let pa32 = self.push32[i][a] as f64;
            if pa32 != 0.0 {
                acc.called_w[p] += pa32 * sum32(&ws.wc);
                for x in 0..m {
                    acc.called[p][x] += pa32 * dot32(&ws.wc, &ws.cur[x * N..(x + 1) * N]);
                }
            }
        }

        let t_p = Instant::now();
        acc.timing[1] += (t_p - t_a).as_secs_f64();
        self.pass_b(a, ws, acc);
        acc.timing[2] += t_p.elapsed().as_secs_f64();
    }

    /// Pass B: X[t][o][c] += sum_b mat_t[b] U_o[a, b, c]. With the overcaller's hand innermost every
    /// caller hand b adds a contiguous row over c, shared by all the triples.
    fn pass_b(&self, a: usize, ws: &mut Workspace, acc: &mut Accumulators) {
        let g = self.g;
        let mw = g.mw.as_ref().unwrap();
        if ws.active.is_empty() {
            return;
        }
        // the caller hands of the active lines, their weights packed, and per order the tensor rows
        // U_o[a, b, .] over the overcaller's hand c
        ws.b_union.clear();
        ws.in_union.iter_mut().for_each(|x| *x = false);
        for &t in &ws.active {
            for &b in &self.call_support[g.triple_pair[t]] {
                if !ws.in_union[b] {
                    ws.in_union[b] = true;
                    ws.b_union.push(b);
                }
            }
        }
        // active triples are grouped by pair, whose triples share the caller's range
        {
            let (active, b_union, mat) = (&ws.active, &ws.b_union, &ws.mat);
            let mut b_pos = [0usize; N];
            for (bi, &b) in b_union.iter().enumerate() {
                b_pos[b] = bi;
            }
            let order: Vec<usize> = (0..active.len()).collect(); // active triples come pair by pair
            ws.mat_packed.pack(&order, |ti| {
                let t = active[ti];
                self.call_support[g.triple_pair[t]].iter().map(|&b| (b_pos[b], mat[t * N + b])).collect()
            });
            // the overcaller's value moved by the shover's and caller's dead-card strength
            let m = g.m;
            ws.shifted_packed.pack(&order, |ti| {
                let t = active[ti];
                let k = g.triples[t].2;
                let (st, al) = (&self.strength[t], &g.alpha[t]);
                let base = al[k] * st[a];
                self.call_support[g.triple_pair[t]]
                    .iter()
                    .map(|&b| (b_pos[b], mat[t * N + b] * (base + al[m + k] * st[b])))
                    .collect()
            });
        }
        for o in 0..SLOTS {
            let (layout, rows) =
                if o < OUTCOMES { (mw.t[g.minor[o]], &ws.mat_packed) } else { (mw.mass, &ws.shifted_packed) };
            for (bi, &b) in ws.b_union.iter().enumerate() {
                ws.pack_b[bi * PAD_N..bi * PAD_N + N].copy_from_slice(&layout[a * NN + b * N..a * NN + b * N + N]);
            }
            gemm(rows, &ws.pack_b, PAD_N, PAD_N, &mut ws.x_row, PAD_N);
            for (ti, &t) in ws.active.iter().enumerate() {
                let dst = &mut acc.x[(t * SLOTS + o) * N..(t * SLOTS + o + 1) * N];
                let src = &ws.x_row[ti * PAD_N..ti * PAD_N + N];
                dst.iter_mut().zip(src).for_each(|(d, x)| *d += *x as f64);
            }
        }
    }
}

fn update(x: &mut [f64], ev: &(Vec<f64>, f64), rate: f64) -> f64 {
    let mut delta: f64 = 0.0;
    for (v, &e) in x.iter_mut().zip(&ev.0) {
        let br = if e > ev.1 { 1.0 } else { 0.0 };
        let d = rate * (br - *v);
        *v += d;
        delta = delta.max(d.abs());
    }
    delta
}

/// Fictitious play from `s`; the average counts as `offset` extra iterations, and with `fresh_over`
/// the overcall strategies jump to the first best response.
#[allow(clippy::too_many_arguments)]
fn fictitious_play(
    g: &Game,
    s: &mut Strategies,
    multiway: bool,
    iterations: usize,
    tol: f64,
    offset: f64,
    fresh_over: bool,
    p: &Params,
    cache: &mut DeadCache,
) {
    for it in 0..iterations {
        let ev = g.evaluate(s, multiway, p.threads, p.mw_min_freq, cache);
        let step = p.averaging / (it as f64 + p.averaging + 1.0 + offset);
        let mut delta: f64 = 0.0;
        for (i, (x, e)) in s.push.iter_mut().zip(&ev.push).enumerate() {
            if !(p.forced_open && i == 0) {
                delta = delta.max(update(x, e, step));
            }
        }
        for (x, e) in s.call.iter_mut().zip(&ev.call) {
            delta = delta.max(update(x, e, step));
        }
        if multiway {
            let rate = if fresh_over && it == 0 { 1.0 } else { step };
            for (x, e) in s.over.iter_mut().zip(&ev.over) {
                let d = update(x, e, rate);
                if rate == step {
                    delta = delta.max(d);
                }
            }
        }
        if it > 50 && delta < tol {
            break;
        }
    }
}

/// Solve without overcalls; with them, continue from that solution as `mbr.pushfold.Solver.solve` does.
pub fn solve(g: &Game, p: &Params) -> (Strategies, Ev) {
    let mut s = g.initial();
    if p.forced_open {
        s.push[0] = vec![1.0; N];
    }
    let mut cache = DeadCache::default();
    fictitious_play(g, &mut s, false, p.hu_iterations, p.tol, 0.0, false, p, &mut cache);
    let multiway = g.mw.is_some() && !g.triples.is_empty();
    if multiway {
        let clean = |x: &mut Vec<f64>| x.iter_mut().for_each(|v| if *v < p.warm_min_freq { *v = 0.0 });
        s.push.iter_mut().for_each(clean);
        s.call.iter_mut().for_each(clean);
        s.over.iter_mut().for_each(|x| x.iter_mut().for_each(|v| *v = 0.0));
        fictitious_play(g, &mut s, true, p.mw_iterations, p.tol, p.warm_weight, true, p, &mut cache);
    }
    // the final EVs with the dead-card equities built for these strategies
    let ev = g.evaluate(&s, multiway, p.threads, p.mw_min_freq, &mut DeadCache::default());
    (s, ev)
}
