/*
 * Monte Carlo preflop all-in equity for all 169x169 starting hand classes.
 *
 * Class index = row * 13 + col on the standard grid (rank 0 = Ace ... 12 = Two):
 * row == col pair, row < col suited, row > col offsuit.
 * Output: 169 lines of 169 equities (win + tie/2) of the row class vs the column class.
 *
 * Usage: equity <samples_per_class_pair> > equity_169.txt
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static inline uint64_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* card = rank * 4 + suit, rank 0 = Two ... 12 = Ace */
static int straight_top(int m) {
    int w = (m << 1) | ((m >> 12) & 1); /* bit 0 = Ace low */
    for (int top = 13; top >= 4; top--)
        if (((w >> (top - 4)) & 31) == 31) return top - 1;
    return -1;
}

static int kickers(int mask, int exclude, int n) {
    int out = 0, taken = 0;
    for (int r = 12; r >= 0 && taken < n; r--) {
        if ((mask >> r & 1) && !(exclude >> r & 1)) {
            out = (out << 4) | r;
            taken++;
        }
    }
    while (taken < n) { out <<= 4; taken++; }
    return out;
}

static int eval7(const int *c) {
    int cnt[13] = {0}, sm[4] = {0}, sc[4] = {0}, mask = 0;
    for (int i = 0; i < 7; i++) {
        int r = c[i] >> 2, s = c[i] & 3;
        cnt[r]++;
        sm[s] |= 1 << r;
        sc[s]++;
        mask |= 1 << r;
    }
    for (int s = 0; s < 4; s++) {
        if (sc[s] >= 5) {
            int st = straight_top(sm[s]);
            if (st >= 0) return (8 << 20) | (st << 16);
            return (5 << 20) | kickers(sm[s], 0, 5);
        }
    }
    int quad = -1, t1 = -1, t2 = -1, p1 = -1, p2 = -1;
    for (int r = 12; r >= 0; r--) {
        if (cnt[r] == 4) { if (quad < 0) quad = r; }
        else if (cnt[r] == 3) { if (t1 < 0) t1 = r; else if (t2 < 0) t2 = r; }
        else if (cnt[r] == 2) { if (p1 < 0) p1 = r; else if (p2 < 0) p2 = r; }
    }
    if (quad >= 0) return (7 << 20) | (quad << 16) | (kickers(mask, 1 << quad, 1) << 12);
    if (t1 >= 0 && (t2 >= 0 || p1 >= 0)) {
        int pp = t2 > p1 ? t2 : p1;
        return (6 << 20) | (t1 << 16) | (pp << 12);
    }
    int st = straight_top(mask);
    if (st >= 0) return (4 << 20) | (st << 16);
    if (t1 >= 0) return (3 << 20) | (t1 << 16) | (kickers(mask, 1 << t1, 2) << 8);
    if (p2 >= 0) return (2 << 20) | (p1 << 16) | (p2 << 12) | (kickers(mask, (1 << p1) | (1 << p2), 1) << 8);
    if (p1 >= 0) return (1 << 20) | (p1 << 16) | (kickers(mask, 1 << p1, 3) << 4);
    return kickers(mask, 0, 5);
}

/* fill combos of a class; returns number of combos */
static int class_combos(int cls, int combos[][2]) {
    int row = cls / 13, col = cls % 13;
    int r1 = 12 - row, r2 = 12 - col; /* grid rank 0 = Ace -> card rank 12 */
    int n = 0;
    if (row == col) {
        for (int s1 = 0; s1 < 4; s1++)
            for (int s2 = s1 + 1; s2 < 4; s2++) { combos[n][0] = r1 * 4 + s1; combos[n][1] = r1 * 4 + s2; n++; }
    } else if (row < col) {
        for (int s = 0; s < 4; s++) { combos[n][0] = r1 * 4 + s; combos[n][1] = r2 * 4 + s; n++; }
    } else {
        for (int s1 = 0; s1 < 4; s1++)
            for (int s2 = 0; s2 < 4; s2++)
                if (s1 != s2) { combos[n][0] = r2 * 4 + s1; combos[n][1] = r1 * 4 + s2; n++; }
    }
    return n;
}

static double matchup(int a, int b, long samples) {
    int ca[12][2], cb[12][2], pairs[144][4], np = 0;
    int na = class_combos(a, ca), nb = class_combos(b, cb);
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++) {
            if (ca[i][0] == cb[j][0] || ca[i][0] == cb[j][1] || ca[i][1] == cb[j][0] || ca[i][1] == cb[j][1]) continue;
            pairs[np][0] = ca[i][0]; pairs[np][1] = ca[i][1]; pairs[np][2] = cb[j][0]; pairs[np][3] = cb[j][1];
            np++;
        }
    if (np == 0) return 0.5;
    long per_pair = samples / np + 1;
    double score = 0;
    long total = 0;
    int deck[52], h1[7], h2[7];
    for (int p = 0; p < np; p++) {
        int nd = 0;
        for (int c = 0; c < 52; c++)
            if (c != pairs[p][0] && c != pairs[p][1] && c != pairs[p][2] && c != pairs[p][3]) deck[nd++] = c;
        h1[0] = pairs[p][0]; h1[1] = pairs[p][1];
        h2[0] = pairs[p][2]; h2[1] = pairs[p][3];
        for (long s = 0; s < per_pair; s++) {
            for (int k = 0; k < 5; k++) {
                int idx = k + (int)(rng() % (uint64_t)(nd - k));
                int t = deck[k]; deck[k] = deck[idx]; deck[idx] = t;
                h1[2 + k] = h2[2 + k] = deck[k];
            }
            int v1 = eval7(h1), v2 = eval7(h2);
            score += v1 > v2 ? 1.0 : (v1 == v2 ? 0.5 : 0.0);
            total++;
        }
    }
    return score / total;
}

int main(int argc, char **argv) {
    long samples = argc > 1 ? atol(argv[1]) : 20000;
    static double eq[169][169];
    for (int a = 0; a < 169; a++)
        for (int b = a; b < 169; b++) {
            double e = matchup(a, b, samples);
            eq[a][b] = e;
            eq[b][a] = 1.0 - e;
        }
    for (int a = 0; a < 169; a++) {
        for (int b = 0; b < 169; b++) printf(b ? " %.5f" : "%.5f", eq[a][b]);
        printf("\n");
    }
    return 0;
}
