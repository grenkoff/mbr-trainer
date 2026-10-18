/*
 * Monte Carlo preflop 3-way all-in outcomes for all 169^3 ordered triples of hand classes.
 *
 * T[x][y][z] = (number of card-disjoint combo triples of classes x, y, z)
 *            * P(the x hand ends best, the y hand second and the z hand worst).
 * Ties are split evenly between the orders they are consistent with. Summing T over the six
 * permutations of (x, y, z) gives the combo triple count, so T carries the card removal weights too.
 *
 * Output: 169^3 little-endian float32 values, x slowest, z fastest.
 *
 * Usage: equity3 <samples_per_class_triple> <output_file>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "eval7.h"

#define N 169

static const int PERMS[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

static float *T;

static void solve_triple(int x, int y, int z, long samples, uint64_t *rs) {
    int cls[3] = {x, y, z};
    int combos[3][12][2], nc[3];
    for (int r = 0; r < 3; r++) nc[r] = class_combos(cls[r], combos[r]);

    /* every card-disjoint combo triple, as 64-bit card masks and card pairs */
    static __thread int trip[1728][3];
    int nt = 0;
    for (int i = 0; i < nc[0]; i++)
        for (int j = 0; j < nc[1]; j++)
            for (int k = 0; k < nc[2]; k++) {
                uint64_t a = (1ULL << combos[0][i][0]) | (1ULL << combos[0][i][1]);
                uint64_t b = (1ULL << combos[1][j][0]) | (1ULL << combos[1][j][1]);
                uint64_t c = (1ULL << combos[2][k][0]) | (1ULL << combos[2][k][1]);
                if ((a & b) || (a & c) || (b & c)) continue;
                trip[nt][0] = i; trip[nt][1] = j; trip[nt][2] = k;
                nt++;
            }

    double order[6] = {0};
    if (nt > 0) {
        int h[3][7];
        for (long s = 0; s < samples; s++) {
            int *t = trip[xorshift(rs) % (uint64_t)nt];
            uint64_t used = 0;
            for (int r = 0; r < 3; r++) {
                h[r][0] = combos[r][t[r]][0];
                h[r][1] = combos[r][t[r]][1];
                used |= (1ULL << h[r][0]) | (1ULL << h[r][1]);
            }
            for (int k = 0; k < 5; k++) {
                int card;
                do card = (int)(xorshift(rs) % 52); while (used >> card & 1);
                used |= 1ULL << card;
                h[0][2 + k] = h[1][2 + k] = h[2][2 + k] = card;
            }
            int v[3] = {eval7(h[0]), eval7(h[1]), eval7(h[2])};
            int ok[6], n_ok = 0;
            for (int p = 0; p < 6; p++) {
                ok[p] = v[PERMS[p][0]] >= v[PERMS[p][1]] && v[PERMS[p][1]] >= v[PERMS[p][2]];
                n_ok += ok[p];
            }
            for (int p = 0; p < 6; p++)
                if (ok[p]) order[p] += 1.0 / n_ok;
        }
    }

    /* permutations of a triple with repeated classes land on the same cell: average them */
    double sum[6] = {0};
    int cnt[6] = {0}, cell[6];
    for (int p = 0; p < 6; p++) {
        cell[p] = (cls[PERMS[p][0]] * N + cls[PERMS[p][1]]) * N + cls[PERMS[p][2]];
        for (int q = 0; q <= p; q++)
            if (cell[q] == cell[p]) {
                sum[q] += nt > 0 ? nt * order[p] / samples : 0.0;
                cnt[q]++;
                break;
            }
    }
    for (int p = 0; p < 6; p++)
        if (cnt[p]) T[cell[p]] = (float)(sum[p] / cnt[p]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <samples_per_triple> <output_file>\n", argv[0]);
        return 1;
    }
    long samples = atol(argv[1]);
    T = calloc((size_t)N * N * N, sizeof(float));
    int done = 0;
#pragma omp parallel
    {
        uint64_t rs = 0x9E3779B97F4A7C15ULL;
#pragma omp for schedule(dynamic, 1)
        for (int x = 0; x < N; x++) {
            rs ^= (uint64_t)(x + 1) * 0xD1B54A32D192ED03ULL;
            for (int y = x; y < N; y++)
                for (int z = y; z < N; z++) solve_triple(x, y, z, samples, &rs);
#pragma omp atomic
            done++;
            if (done % 10 == 0) fprintf(stderr, "\r%d/%d", done, N);
        }
    }
    fprintf(stderr, "\n");
    FILE *f = fopen(argv[2], "wb");
    if (!f || fwrite(T, sizeof(float), (size_t)N * N * N, f) != (size_t)N * N * N) {
        perror(argv[2]);
        return 1;
    }
    fclose(f);
    return 0;
}
