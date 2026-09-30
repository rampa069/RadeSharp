/* Deterministic channel model for golden vectors: complex IQ (f32 pairs) in on
   stdin, impaired IQ out on stdout. Kept simple and seedable so the C# tests can
   store its output as a fixture rather than reimplementing it.

   usage: chan <gain> <foff_hz> <noise_rms> <lead_zeros> <seed> */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t s;
static double urand(void)
{
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return ((s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static double grand(void) { return sqrt(-2.0 * log(urand())) * cos(2.0 * M_PI * urand()); }

int main(int argc, char *argv[])
{
    if (argc != 6) { fprintf(stderr, "usage: chan gain foff_hz noise_rms lead_zeros seed\n"); return 1; }
    double gain = atof(argv[1]), foff = atof(argv[2]), nrms = atof(argv[3]);
    long lead = atol(argv[4]);
    s = strtoull(argv[5], NULL, 10) * 2654435761ULL + 1;
    float x[2];
    long n = 0;
    for (long i = 0; i < lead; i++) {
        float y[2] = { (float)(nrms * grand() / sqrt(2.0)), (float)(nrms * grand() / sqrt(2.0)) };
        fwrite(y, sizeof(float), 2, stdout);
    }
    while (fread(x, sizeof(float), 2, stdin) == 2) {
        double ph = 2.0 * M_PI * foff * n++ / 8000.0;
        double c = cos(ph), sn = sin(ph);
        double re = gain * (x[0] * c - x[1] * sn), im = gain * (x[0] * sn + x[1] * c);
        float y[2] = { (float)(re + nrms * grand() / sqrt(2.0)), (float)(im + nrms * grand() / sqrt(2.0)) };
        fwrite(y, sizeof(float), 2, stdout);
    }
    return 0;
}
