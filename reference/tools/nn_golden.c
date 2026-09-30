/* Golden vectors for the neural networks alone (no modem DSP): runs the RADE
   V1/V2 encoders, decoders and FrameSyncNet on a deterministic input sequence.

   usage: nn_golden <out_dir>
   writes nn_in.f32 (84/frame), nn_v1_lat.f32 (80), nn_v1_dec.f32 (84),
          nn_v2_lat.f32 (56), nn_v2_dec.f32 (84), nn_v2_sync.f32 (1) */

#include <math.h>
#include <stdio.h>
#include "rade_enc.h"
#include "rade_dec.h"
#include "rade_enc_v2.h"
#include "rade_dec_v2.h"
#include "rade_sync.h"
#include "rade_enc_data.h"
#include "rade_dec_data.h"
#include "rade_enc_v2_data.h"
#include "rade_dec_v2_data.h"
#include "rade_sync_data.h"

#define NFRAMES 60
#define NIN 84

static FILE *open_out(const char *dir, const char *name)
{
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s", dir, name);
    return fopen(p, "wb");
}

int main(int argc, char *argv[])
{
    if (argc != 2) return 1;
    static RADEEnc e1; static RADEDec d1; static RADEEncV2 e2; static RADEDecV2 d2; static RADESync s2;
    static RADEEncState e1s; static RADEDecState d1s; static RADEEncV2State e2s; static RADEDecV2State d2s;
    if (init_radeenc(&e1, radeenc_arrays, NIN) || init_radedec(&d1, radedec_arrays, NIN) ||
        init_radeencv2(&e2, radeencv2_arrays) || init_radedecv2(&d2, radedecv2_arrays) ||
        init_radesync(&s2, radesync_arrays)) { fprintf(stderr, "init failed\n"); return 1; }
    rade_init_encoder(&e1s); rade_init_decoder(&d1s); rade_init_encoder_v2(&e2s); rade_init_decoder_v2(&d2s);

    FILE *fin = open_out(argv[1], "nn_in.f32"), *f1l = open_out(argv[1], "nn_v1_lat.f32"),
         *f1d = open_out(argv[1], "nn_v1_dec.f32"), *f2l = open_out(argv[1], "nn_v2_lat.f32"),
         *f2d = open_out(argv[1], "nn_v2_dec.f32"), *f2s = open_out(argv[1], "nn_v2_sync.f32");
    for (int f = 0; f < NFRAMES; f++) {
        float in[NIN], l1[80], o1[NIN], l2[56], o2[84];
        for (int i = 0; i < NIN; i++) in[i] = (float)(1.3 * sin(0.37 * i + 0.11 * f + 0.05 * i * f));
        rade_core_encoder(&e1s, &e1, l1, in, 0, 3);
        rade_core_decoder(&d1s, &d1, o1, l1, 0);
        rade_core_encoder_v2(&e2s, &e2, l2, in, 0);
        rade_core_decoder_v2(&d2s, &d2, o2, l2, 0);
        float s = rade_frame_sync(&s2, l2, 0);
        fwrite(in, 4, NIN, fin); fwrite(l1, 4, 80, f1l); fwrite(o1, 4, NIN, f1d);
        fwrite(l2, 4, 56, f2l); fwrite(o2, 4, 84, f2d); fwrite(&s, 4, 1, f2s);
    }
    fclose(fin); fclose(f1l); fclose(f1d); fclose(f2l); fclose(f2d); fclose(f2s);
    return 0;
}
