/* Platform-independent rand()/srand() for the reference build.

   rade_acq.c calls libc rand() (unseeded) to pick correlation-grid cells to
   refresh, so its output depends on the libc. Linking this into librade makes
   the golden vectors identical on every OS. The algorithm is the Park-Miller
   "minimal standard" generator used by macOS libc (seed 1), which RadeSharp's
   CRand reproduces. */

#include <stdlib.h>

static unsigned long next = 1;

int rand(void)
{
    long hi, lo, x;
    if (next == 0) next = 123459876;
    hi = next / 127773;
    lo = next % 127773;
    x = 16807 * lo - 2836 * hi;
    if (x < 0) x += 0x7fffffff;
    next = x;
    return (int)(x % ((unsigned long)RAND_MAX + 1));
}

void srand(unsigned seed) { next = seed; }
