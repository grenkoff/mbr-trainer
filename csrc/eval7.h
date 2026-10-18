/* Shared helpers for the equity generators: xorshift RNG, 7-card evaluator, hand class combos. */
#ifndef EVAL7_H
#define EVAL7_H

#include <stdint.h>

static inline uint64_t xorshift(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
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

#endif
