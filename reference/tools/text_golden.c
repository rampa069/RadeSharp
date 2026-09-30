/* Golden vectors for the FreeDV EOO reliable-text codec (rade_text + codec2 LDPC).
   For each callsign: generate the 180 EOO floats, perturb them deterministically
   (gain + noise of increasing strength), decode, record the result.

   usage: text_golden <out_dir>
   writes text_tx.f32, text_rx_in.f32 (NCASES x 180 floats) and text_rx.txt
   (one line per case: decoded callsign or "-" when nothing valid decoded). */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "rade_text.h"

#define NFLOATS 180   /* rade_n_eoo_bits() for RADE V1 */

static char last[64];
static void on_rx(rade_text_t rt, const char *txt, int len, void *state)
{
    (void)rt; (void)state;
    snprintf(last, sizeof(last), "%.*s", len, txt);
}

static uint64_t s = 88172645463325252ULL;
static double urand(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return ((s >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double grand(void) { return sqrt(-2.0 * log(urand())) * cos(2.0 * M_PI * urand()); }

int main(int argc, char *argv[])
{
    if (argc != 2) return 1;
    const char *calls[] = {"EA5IUE", "N9WAR", "vk5dgr", "G8SEZ/P", "AB1CDEFGH", "K0PFX", "", "2E0ABC", "W1AW", "JH0VEQ"};
    const float noise[] = {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.2f, 1.5f};
    char path[1024];
    snprintf(path, sizeof(path), "%s/text_tx.f32", argv[1]); FILE *ftx = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/text_rx_in.f32", argv[1]); FILE *fin = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/text_rx.txt", argv[1]); FILE *fo = fopen(path, "w");
    rade_text_t tx = rade_text_create(), rx = rade_text_create();
    rade_text_set_rx_callback(rx, on_rx, NULL);
    for (unsigned c = 0; c < sizeof(calls) / sizeof(calls[0]); c++) {
        float syms[NFLOATS], in[NFLOATS];
        rade_text_generate_tx_string(tx, calls[c], (int)strlen(calls[c]), syms, NFLOATS);
        fwrite(syms, sizeof(float), NFLOATS, ftx);
        for (unsigned k = 0; k < sizeof(noise) / sizeof(noise[0]); k++) {
            for (int i = 0; i < NFLOATS; i++) in[i] = (float)(0.6 * syms[i] + noise[k] * grand());
            fwrite(in, sizeof(float), NFLOATS, fin);
            last[0] = 0;
            strcpy(last, "-");
            rade_text_rx(rx, in, NFLOATS / 2);
            fprintf(fo, "%s\n", last);
        }
    }
    fclose(ftx); fclose(fin); fclose(fo);
    return 0;
}
