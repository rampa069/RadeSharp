# Updating RadeSharp to new upstream versions

RadeSharp is a line-by-line port. It must stay **bit-exact** with a scalar,
no-FMA build of the C code it tracks, and the test suite enforces that. Updating
therefore means repeating the port for whatever changed, then proving parity again
with regenerated golden vectors.

## What is tracked

| Upstream | Pin (`reference/pins.cmake`) | Ported into |
|---|---|---|
| [freedv/rade_c](https://github.com/freedv/rade_c) | `RADE_C_SHA` | `src/RadeSharp` (API, DSP, V1, V2, neural nets, weights) |
| [xiph/opus](https://github.com/xiph/opus) `dnn/` + parts of `celt/` | `OPUS_SHA` (must equal rade_c's `cmake/BuildOpus.cmake`) | `src/RadeSharp/Nnet`, `src/RadeSharp/Vocoder` |
| freedv-gui `rade_text.c` + codec2 v1.2.0 LDPC slice, as vendored by Zeus | `FREEDV_TEXT_SHA` | `src/RadeSharp.Text` |

`scripts/port-map.tsv` maps every upstream file to its C# counterpart and says how
it is maintained:

- `port`: hand-ported. Changes must be re-ported by reading the C diff.
- `gen`: produced by a script (weights, model bindings, `phi0` table). Don't edit by hand.
- `ref`: only used by the reference build, the golden vectors or the docs.
- `skip`: not needed by the port.

## Procedure

Directory layout assumed by the scripts (all overridable with env vars):

```
Radae/
├── rade_c/            upstream checkout (RADE_C_DIR)
├── rade_ref-build/    scalar reference build (BUILD_DIR). Its path must NOT contain '#'
└── radae_c#/          this repo
../zeus/native/radae/vendor/freedv_text   (FREEDV_TEXT_DIR)
```

1. **See what changed.**
   ```
   git -C ../rade_c fetch
   scripts/upstream-diff.sh rade_c origin/main
   ```
   The script lists the commits, then every changed file with its C# target and kind.
   `UNMAPPED` files need a decision. Add them to `port-map.tsv` before you go on.
   Do the same for Opus if rade_c's `cmake/BuildOpus.cmake` moved the Opus pin
   (`OPUS_DIR=/path/to/opus scripts/upstream-diff.sh opus <new-sha>`).

2. **Open a beads issue** for the update (`bd create ...`). Give each non-trivial
   C diff its own child issue.

3. **Move the pins.** Check out the new commit in `../rade_c` and update
   `RADE_C_SHA` (and `OPUS_SHA` if needed) in `reference/pins.cmake`. If upstream
   added `.c` files, source them in `reference/CMakeLists.txt` too (see the
   `port-map` entries for `CMakeLists.txt`).

4. **Port the `port` diffs.** Work file by file with `git -C ../rade_c diff
   <old> <new> -- src/<file>`. Follow the numerics rules below: they are what keep
   the port bit-exact.

5. **Regenerate everything derived.**
   ```
   scripts/regenerate.sh
   ```
   This rebuilds the reference and re-exports the weight blobs and
   `lpcnet_tables.bin`. It regenerates `Models/Generated/*.g.cs` and `Phi0.cs`,
   rewrites `tests/fixtures`, and runs the tests.

6. **Read the failures in order.**
   - `ModelShapeTests` failing: the network architecture changed (layer sizes or
     new layers). Update the constants and stage loops in `Core/RadeCoreNets.cs`.
     `git diff src/RadeSharp/Models/Generated` shows exactly what changed.
   - `NnetTests` failing: the neural core diverged. Check `Nnet/*` against the new
     Opus `nnet.c` / `nnet_arch.h` / `vec.h`, and `Core/RadeCoreNets.cs` against
     `rade_{enc,dec}{,_v2}.c` and `rade_sync.c`.
   - `ApiTraceTests` failing: the modem diverged. The message names the first
     `rade_rx()` call and field that differs (nin, sync, freq offset, SNR, stats,
     data symbol, EOO), which narrows it to the acquisition, the tracking loop or
     the demodulator.
   - `VocoderTests` / `TextTests` / `CliTests`: FARGAN / LPCNet, the EOO text
     codec, or the WAV tools, respectively.

7. **Check the API.** Diff `src/rade_api.h`: new functions, flags or struct fields
   go into `RadeApi.cs` with the same names, and `VERSION` / `VERSION_MINOR` get
   copied. Expose anything useful through `RadeModem` / `RadeVoiceModem` too, and
   update `docs/API.md`.

8. **Update the docs:** `README.md`, the pin table above, and `docs/API.md` if the
   behaviour changed.

9. **Benchmark** (`dotnet run -c Release --project tools/RadeSharp.Bench`) and
   compare with the previous numbers in the commit message. A big regression
   usually means a hot-path allocation crept in.

10. **Zeus:** bump the package/project reference and run its FreeDV tests.

### Checklist

```
[ ] upstream-diff.sh reviewed, no UNMAPPED files
[ ] pins.cmake updated (RADE_C_SHA / OPUS_SHA / FREEDV_TEXT_SHA)
[ ] every `port` diff re-ported (one beads issue each)
[ ] regenerate.sh clean: all tests green, generated files committed
[ ] RadeApi.cs matches rade_api.h (names, flags, VERSION)
[ ] README / docs/API.md / UPDATING.md pin table updated
[ ] benchmark RTF recorded
```

## Numerics rules (how bit-exactness is kept)

The reference is built scalar (`--disable-intrinsics --disable-rtcd -DDISABLE_NEON`)
with `-ffp-contract=off`. The .NET JIT never fuses multiply-adds, so the same
operation order gives the same bits. On top of that:

1. **Keep C's promotions.** A double literal (`0.5`, `M_PI`, `1.0/x`), a double libm
   call (`sqrt`, `exp`, `floor`, `pow`) or a `double` variable turns the whole
   sub-expression into double math. Write it as an explicit
   `(float)(… double math …)`. `float x = 2.0f * M_PI * f / Fs;` becomes
   `(float)(2.0f * Math.PI * f / Fs)`. `M_PI` cast to float is `(float)Math.PI`.
2. **sin/cos.** When a C function computes `cosf(x)` and `sinf(x)` of the same
   argument (including through inlined helpers like `rade_cexp`), clang and gcc
   fuse them into `sincosf`, which rounds differently from separate calls on macOS.
   Use `MathF.SinCos` there, and `MathF.Sin` / `MathF.Cos` where C computes only
   one. To check, run `nm` on the C object: `___sincosf_stret` vs `_sinf` / `_cosf`.
3. **pow → exp2/exp10.** clang lowers `pow(2, x)` to `exp2` and `pow(10, x)` to
   `__exp10`. .NET has no libm `exp2`/`exp10` (`Math.Pow` differs in the last double
   ulp for ~0.2% of inputs). Every current use rounds the result to an int or a
   float, so the output is identical in practice. Document any new use the same way.
4. **Summation order is semantics.** Keep loop order, the pairwise
   `rade_cdot_comp` recursion, the `(a+b)+c` grouping, and the `x += a*b` versus
   `x = x + (a*b + c*d)` forms of the Opus kernels.
5. **Integer conversions.** `(int)` truncates in both languages. C's `(int8)` wraps
   and so does `unchecked((sbyte)…)`. Float→int overflow saturates in .NET, which
   is what C does on ARM.
6. **`rand()`.** `rade_acq.c` uses libc `rand()`. The reference links
   `reference/tools/crand.c` (Park–Miller, macOS libc) so goldens are the same on
   every OS, and `Dsp/CRand.cs` implements the same generator per context.
7. **Tables come from C.** The weights, FFT twiddles/bitrev, window and DCT table
   are exported from the linked C objects (`reference/tools/dump_weights.c`), never
   re-derived in C#.

## Known deliberate deviations

- `rand()` state is per context instead of process-global.
- `RadeText` guards an out-of-bounds read that the C code does when stats are
  enabled and `symSize` exceeds the payload.
- `RadeVoiceModem` (Zeus shim equivalent) copies the shim's choices on purpose:
  re-priming reuses the FARGAN state, the int16 rounding uses `floorf`, and
  `rade_text_rx` gets `n_eoo_bits` (not `/2`).
