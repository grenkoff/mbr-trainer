/*
 * Monte Carlo preflop all-in equity for all 169x169 starting hand classes.
 *
 * Class index = row * 13 + col on the standard grid (rank 0 = Ace ... 12 = Two):
 * row == col pair, row < col suited, row > col offsuit.
 * Output: 169 lines of 169 equities (win + tie/2) of the row class vs the column class, or with
 * "ties" the probabilities that the two hands tie.
 *
 * Usage: equity <samples_per_class_pair> [ties] > equity_169.txt
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eval7.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static inline uint64_t rng(void) { return xorshift(&rng_state); }

static double matchup(int a, int b, long samples, double *tie) {
    int ca[12][2], cb[12][2], pairs[144][4], np = 0;
    int na = class_combos(a, ca), nb = class_combos(b, cb);
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++) {
            if (ca[i][0] == cb[j][0] || ca[i][0] == cb[j][1] || ca[i][1] == cb[j][0] || ca[i][1] == cb[j][1]) continue;
            pairs[np][0] = ca[i][0]; pairs[np][1] = ca[i][1]; pairs[np][2] = cb[j][0]; pairs[np][3] = cb[j][1];
            np++;
        }
    *tie = 0;
    if (np == 0) return 0.5;
    long per_pair = samples / np + 1;
    double score = 0;
    long total = 0, ties = 0;
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
            ties += v1 == v2;
            total++;
        }
    }
    *tie = (double)ties / total;
    return score / total;
}

int main(int argc, char **argv) {
    long samples = argc > 1 ? atol(argv[1]) : 20000;
    int want_ties = argc > 2 && strcmp(argv[2], "ties") == 0;
    static double eq[169][169];
    for (int a = 0; a < 169; a++)
        for (int b = a; b < 169; b++) {
            double tie, e = matchup(a, b, samples, &tie);
            eq[a][b] = want_ties ? tie : e;
            eq[b][a] = want_ties ? tie : 1.0 - e;
        }
    for (int a = 0; a < 169; a++) {
        for (int b = 0; b < 169; b++) printf(b ? " %.5f" : "%.5f", eq[a][b]);
        printf("\n");
    }
    return 0;
}
