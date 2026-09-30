/* Golden-vector generator: drives the public rade_api.h exactly like an SDR
   integration would and records everything observable per call, so the C#
   port can be compared call by call.

   tx: rade_trace tx [--v2] <features.f32> <out.iq>
       V1: EOO bits = deterministic +/-1 pattern. V2: data symbol per frame from
       a deterministic pattern. Writes the frames, the EOO frame, then one
       EOO-length block of zeros (same as radae_tx).
   rx: rade_trace rx [--v2] [--agc 0|1] <in.iq> <features_out.f32> <trace.bin>
       trace.bin: one rx_record per rade_rx() call (see struct below). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rade_api.h"

typedef struct {
    int32_t nin, nout, has_eoo, sync;
    float freq_offset, snr_db, data_symbol;
    int32_t st_sync;
    float st_delta_hat, st_delta_hat_g, st_freq_offset, st_gain, st_snr_est;
    float eoo_bits_sum;   /* V1: sum of |eoo bits| * sign pattern -> cheap checksum */
} rx_record;

static float pattern(int i) { return ((i * 7 + (i >> 3)) % 3) ? 1.0f : -1.0f; }

int main(int argc, char *argv[])
{
    int flags = RADE_VERBOSE_0, a = 2, agc = -1;
    if (argc < 2) return 1;
    int tx = strcmp(argv[1], "tx") == 0;
    while (a < argc && argv[a][0] == '-' && argv[a][1] == '-') {
        if (strcmp(argv[a], "--v2") == 0) flags |= RADE_MODE_V2;
        else if (strcmp(argv[a], "--agc") == 0) agc = atoi(argv[++a]);
        a++;
    }
    rade_initialize();
    struct rade *r = rade_open("", flags);
    if (tx) {
        FILE *fi = fopen(argv[a], "rb"), *fo = fopen(argv[a + 1], "wb");
        int nf = rade_n_features_in_out(r), ntx = rade_n_tx_out(r), neoo = rade_n_tx_eoo_out(r);
        float *feat = malloc(sizeof(float) * nf);
        RADE_COMP *out = malloc(sizeof(RADE_COMP) * (ntx > neoo ? ntx : neoo));
        int nbits = rade_n_eoo_bits(r);
        if (nbits) {
            float *bits = malloc(sizeof(float) * nbits);
            for (int i = 0; i < nbits; i++) bits[i] = pattern(i);
            rade_tx_set_eoo_bits(r, bits);
            free(bits);
        }
        int frame = 0;
        while (fread(feat, sizeof(float), nf, fi) == (size_t)nf) {
            rade_tx_set_data_symbol(r, pattern(frame++));
            int n = rade_tx(r, out, feat);
            fwrite(out, sizeof(RADE_COMP), n, fo);
        }
        int n = rade_tx_eoo(r, out);
        fwrite(out, sizeof(RADE_COMP), n, fo);
        memset(out, 0, sizeof(RADE_COMP) * n);
        fwrite(out, sizeof(RADE_COMP), n, fo);
        fclose(fi); fclose(fo);
    } else {
        if (agc >= 0) rade_rx_set_agc(r, agc);
        FILE *fi = fopen(argv[a], "rb"), *fo = fopen(argv[a + 1], "wb"), *ft = fopen(argv[a + 2], "wb");
        int nf = rade_n_features_in_out(r), nbits = rade_n_eoo_bits(r);
        RADE_COMP *in = malloc(sizeof(RADE_COMP) * rade_nin_max(r));
        float *feat = malloc(sizeof(float) * nf);
        float *eoo = malloc(sizeof(float) * (nbits ? nbits : 1));
        for (;;) {
            rx_record rec;
            memset(&rec, 0, sizeof(rec));
            rec.nin = rade_nin(r);
            if (fread(in, sizeof(RADE_COMP), rec.nin, fi) != (size_t)rec.nin) break;
            int has_eoo = 0;
            rec.nout = rade_rx(r, feat, &has_eoo, nbits ? eoo : NULL, in);
            rec.has_eoo = has_eoo;
            rec.sync = rade_sync(r);
            rec.freq_offset = rade_freq_offset(r);
            rec.snr_db = rade_snrdB_3k_est(r);
            rec.data_symbol = rade_rx_get_data_symbol(r);
            struct rade_stats st;
            rade_get_stats(r, &st);
            rec.st_sync = st.sync; rec.st_delta_hat = st.delta_hat; rec.st_delta_hat_g = st.delta_hat_g;
            rec.st_freq_offset = st.freq_offset; rec.st_gain = st.gain; rec.st_snr_est = st.snr_est;
            if (has_eoo && nbits) for (int i = 0; i < nbits; i++) rec.eoo_bits_sum += eoo[i] * pattern(i);
            if (rec.nout > 0) fwrite(feat, sizeof(float), rec.nout, fo);
            fwrite(&rec, sizeof(rec), 1, ft);
        }
        fclose(fi); fclose(fo); fclose(ft);
    }
    rade_close(r);
    rade_finalize();
    return 0;
}
