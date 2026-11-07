/*
 * How a dead card changes heads-up preflop all-in results, for all 169x169 hand class pairs.
 *
 * Cards held by players who folded are dead: they cannot come on the board. A card dealt at random
 * changes nothing on average, but folded hands hold some ranks more often than others. For a card c
 * left in the deck, d(c) = E[result | c is dead] - E[result] (c out of the deck is the same as the
 * board drawn from the full deck conditioned on c not being on it). The output is, for each rank r,
 *
 *   D[a][b][r] = (sum of d(c) over the rank r cards c left in the deck) / 4,
 *
 * so that the result with dead cards is about E[a][b] + sum over r of D[a][b][r] * (expected number of
 * rank r cards among the dead ones - their number for random dead cards), to first order.
 *
 * Results: the equity of a vs b (win + tie/2), then the tie probability. Output: float32, the equity
 * part [a][b][r] and then the tie part, grid rank order (0 = Ace ... 12 = Two) for r.
 *
 * Usage: dead <samples_per_class_pair> <output_file>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "eval7.h"

#define N 169
#define R 13

static float *DE, *DT;

static void matchup(int a, int b, long samples, uint64_t *rs) {
    int ca[12][2], cb[12][2], pairs[144][4], np = 0;
    int na = class_combos(a, ca), nb = class_combos(b, cb);
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++) {
            if (ca[i][0] == cb[j][0] || ca[i][0] == cb[j][1] || ca[i][1] == cb[j][0] || ca[i][1] == cb[j][1]) continue;
            pairs[np][0] = ca[i][0]; pairs[np][1] = ca[i][1]; pairs[np][2] = cb[j][0]; pairs[np][3] = cb[j][1];
            np++;
        }
    if (np == 0) return;
    long per_pair = samples / np + 1, total = 0;
    /* sums over the samples of result * w_r and of w_r, w_r = (5 * left_r - 48 * on_board_r) / 43 / 4:
       its mean is 0 and result * w_r averages to D[r] (the centred form has less noise) */
    double eq_sum = 0, tie_sum = 0, w_eq[R] = {0}, w_tie[R] = {0}, w_sum[R] = {0};
    int deck[52], h1[7], h2[7];
    for (int p = 0; p < np; p++) {
        int nd = 0, left[R];
        for (int r = 0; r < R; r++) left[r] = 4;
        for (int k = 0; k < 4; k++) left[12 - pairs[p][k] / 4]--;
        for (int c = 0; c < 52; c++)
            if (c != pairs[p][0] && c != pairs[p][1] && c != pairs[p][2] && c != pairs[p][3]) deck[nd++] = c;
        h1[0] = pairs[p][0]; h1[1] = pairs[p][1];
        h2[0] = pairs[p][2]; h2[1] = pairs[p][3];
        for (long s = 0; s < per_pair; s++) {
            int on_board[R] = {0};
            for (int k = 0; k < 5; k++) {
                int idx = k + (int)(xorshift(rs) % (uint64_t)(nd - k));
                int t = deck[k]; deck[k] = deck[idx]; deck[idx] = t;
                h1[2 + k] = h2[2 + k] = deck[k];
                on_board[12 - deck[k] / 4]++;
            }
            int v1 = eval7(h1), v2 = eval7(h2);
            double eq = v1 > v2 ? 1.0 : (v1 == v2 ? 0.5 : 0.0), tie = v1 == v2;
            eq_sum += eq;
            tie_sum += tie;
            for (int r = 0; r < R; r++) {
                double w = (5.0 * left[r] - 48.0 * on_board[r]) / 43.0 / 4.0;
                w_eq[r] += eq * w;
                w_tie[r] += tie * w;
                w_sum[r] += w;
            }
            total++;
        }
    }
    for (int r = 0; r < R; r++) {
        double de = a == b ? 0.0 : (w_eq[r] - eq_sum / total * w_sum[r]) / total; /* 0 by symmetry */
        double dt = (w_tie[r] - tie_sum / total * w_sum[r]) / total;
        DE[((size_t)a * N + b) * R + r] = (float)de;
        DE[((size_t)b * N + a) * R + r] = (float)-de;
        DT[((size_t)a * N + b) * R + r] = (float)dt;
        DT[((size_t)b * N + a) * R + r] = (float)dt;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <samples_per_class_pair> <output_file>\n", argv[0]);
        return 1;
    }
    long samples = atol(argv[1]);
    size_t cells = (size_t)N * N * R;
    DE = calloc(2 * cells, sizeof(float));
    DT = DE + cells;
#pragma omp parallel
    {
        uint64_t rs = 0x9E3779B97F4A7C15ULL;
#pragma omp for schedule(dynamic, 1)
        for (int a = 0; a < N; a++) {
            rs ^= (uint64_t)(a + 1) * 0xD1B54A32D192ED03ULL;
            for (int b = a; b < N; b++) matchup(a, b, samples, &rs);
        }
    }
    FILE *f = fopen(argv[2], "wb");
    if (!f || fwrite(DE, sizeof(float), 2 * cells, f) != 2 * cells) {
        perror(argv[2]);
        return 1;
    }
    fclose(f);
    return 0;
}
