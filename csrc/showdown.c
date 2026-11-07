/* Every way a partial board can run out between known hands, for the all-in adjusted results (mbr/showdown.py).
   Built as a shared library: gcc -O3 -march=native -shared -fPIC -o build/libshowdown.so csrc/showdown.c */
#include "eval7.h"

/* For player i, bits 4i..4i+3: how many players have a strictly better hand on this full board. */
static uint64_t signature(int n, const int *hole, const int *full) {
    int cards[7], strength[9];
    for (int i = 0; i < n; i++) {
        cards[0] = hole[2 * i];
        cards[1] = hole[2 * i + 1];
        for (int k = 0; k < 5; k++) cards[2 + k] = full[k];
        strength[i] = eval7(cards);
    }
    uint64_t sig = 0;
    for (int i = 0; i < n; i++) {
        uint64_t better = 0;
        for (int j = 0; j < n; j++) better += strength[j] > strength[i];
        sig |= better << (4 * i);
    }
    return sig;
}

/* n players' hole cards (2 per player), nb board cards; writes the signature of every completion of the board,
   or of `samples` random ones if samples > 0. Returns the number of boards. */
long showdown_sigs(int n, const int *hole, int nb, const int *board, long samples, uint64_t seed, uint64_t *out) {
    int used[52] = {0}, deck[52], nd = 0, need = 5 - nb, idx[5];
    for (int i = 0; i < 2 * n; i++) used[hole[i]] = 1;
    for (int i = 0; i < nb; i++) used[board[i]] = 1;
    for (int c = 0; c < 52; c++)
        if (!used[c]) deck[nd++] = c;
    int full[5];
    for (int k = 0; k < nb; k++) full[k] = board[k];
    if (samples > 0) {
        for (long t = 0; t < samples; t++) {
            for (int k = 0; k < need; k++) { /* a partial Fisher-Yates shuffle draws the missing cards */
                int j = k + (int)(xorshift(&seed) % (uint64_t)(nd - k)), c = deck[j];
                deck[j] = deck[k];
                deck[k] = c;
                full[nb + k] = c;
            }
            out[t] = signature(n, hole, full);
        }
        return samples;
    }
    for (int k = 0; k < need; k++) idx[k] = k;
    long count = 0;
    for (;;) {
        for (int k = 0; k < need; k++) full[nb + k] = deck[idx[k]];
        out[count++] = signature(n, hole, full);
        /* next combination of `need` cards out of the deck */
        int k = need - 1;
        while (k >= 0 && idx[k] == nd - need + k) k--;
        if (k < 0) break;
        idx[k]++;
        for (int j = k + 1; j < need; j++) idx[j] = idx[j - 1] + 1;
    }
    return count;
}
