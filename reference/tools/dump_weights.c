/* Dumps the weight tables linked into the reference build as Opus "DNNw"
   weight blobs (the format parsed by Opus parse_weights()), one file per model.
   Dumping the linked arrays -- rather than parsing the generated C -- gives
   exactly what the C runtime uses, e.g. Opus' FARGAN/PitchDNN tables are built
   with DISABLE_DEBUG_FLOAT so they carry only the int8 weights.

   usage: dump_weights <out_dir> */

#include <stdio.h>
#include <string.h>
#include "nnet.h"
#include "rade_core.h"
#include "rade_v2_core.h"
/* celt's kiss_fft.h, not rade_c's own copy that comes first on the include path */
#include "celt/kiss_fft.h"
#define WINDOW_SIZE 320   /* dnn/freq.h */
#define OVERLAP_SIZE 160
#define NB_BANDS 18

/* LPCNet analysis tables from Opus dnn/lpcnet_tables.c */
extern const kiss_fft_state kfft;
extern const float half_window[];
extern const float dct_table[];

static int write_weights(const char *dir, const char *file, const WeightArray *list)
{
    char path[1024];
    unsigned char zeros[WEIGHT_BLOCK_SIZE] = {0};
    int i;
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    for (i = 0; list[i].name != NULL; i++) {
        WeightHead h;
        memset(&h, 0, sizeof(h));
        if (strlen(list[i].name) >= sizeof(h.name)) {
            fprintf(stderr, "name too long: %s\n", list[i].name);
            return 1;
        }
        memcpy(h.head, "DNNw", 4);
        h.version = WEIGHT_BLOB_VERSION;
        h.type = list[i].type;
        h.size = list[i].size;
        h.block_size = (h.size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;
        strncpy(h.name, list[i].name, sizeof(h.name) - 1);
        fwrite(&h, 1, WEIGHT_BLOCK_SIZE, f);
        fwrite(list[i].data, 1, h.size, f);
        fwrite(zeros, 1, h.block_size - h.size, f);
    }
    fclose(f);
    fprintf(stderr, "%s: %d arrays\n", path, i);
    return 0;
}

/* Exported with the same "DNNw" container so the C# side needs one parser. */
static int write_tables(const char *dir)
{
    static int bitrev[WINDOW_SIZE];
    static float scale[1];
    static int factors[2 * MAXFACTORS];
    for (int i = 0; i < kfft.nfft; i++) bitrev[i] = kfft.bitrev[i];
    for (int i = 0; i < 2 * MAXFACTORS; i++) factors[i] = kfft.factors[i];
    scale[0] = kfft.scale;
    const WeightArray tables[] = {
        {"kfft_twiddles", WEIGHT_TYPE_float, (int)sizeof(kiss_twiddle_cpx) * WINDOW_SIZE, kfft.twiddles},
        {"kfft_bitrev", WEIGHT_TYPE_int, (int)sizeof(int) * WINDOW_SIZE, bitrev},
        {"kfft_factors", WEIGHT_TYPE_int, (int)sizeof(factors), factors},
        {"kfft_scale", WEIGHT_TYPE_float, (int)sizeof(scale), scale},
        {"half_window", WEIGHT_TYPE_float, (int)sizeof(float) * OVERLAP_SIZE, half_window},
        {"dct_table", WEIGHT_TYPE_float, (int)sizeof(float) * NB_BANDS * NB_BANDS, dct_table},
        {NULL, 0, 0, NULL}
    };
    if (kfft.nfft != WINDOW_SIZE || kfft.shift > 0) { fprintf(stderr, "unexpected kfft config\n"); return 1; }
    return write_weights(dir, "lpcnet_tables.bin", tables);
}

int main(int argc, char *argv[])
{
    if (argc != 2) { fprintf(stderr, "usage: %s <out_dir>\n", argv[0]); return 1; }
    const char *d = argv[1];
    return write_weights(d, "rade_v1_enc.bin", radeenc_arrays)
         | write_weights(d, "rade_v1_dec.bin", radedec_arrays)
         | write_weights(d, "rade_v2_enc.bin", radeencv2_arrays)
         | write_weights(d, "rade_v2_dec.bin", radedecv2_arrays)
         | write_weights(d, "rade_v2_sync.bin", radesync_arrays)
         | write_weights(d, "fargan.bin", fargan_arrays)
         | write_weights(d, "pitchdnn.bin", pitchdnn_arrays)
         | write_tables(d);
}
