# RadeSharp API

## `RadeApi`: one-to-one with `rade_api.h`

`using static RadeSharp.RadeApi;` lets C code port almost unchanged. Semantics are
those of [RadeAPIUse.md](https://github.com/freedv/rade_c/blob/main/RadeAPIUse.md).

| C (`rade_api.h`) | C# (`RadeApi`) |
|---|---|
| `RADE_COMP` | `RadeComp` (`Real`, `Imag`; same memory layout) |
| `struct rade *` | `Rade` (opaque handle) |
| `struct rade_stats` | `RadeStats` (same field names) |
| `RADE_MODEM_SAMPLE_RATE`, `RADE_SPEECH_SAMPLE_RATE`, `RADE_INT16_SCALE` | same names, `const` |
| `RADE_USE_C_ENCODER` … `RADE_NO_TX_BPF` | same names, `const int` |
| `void rade_initialize(void)` / `rade_finalize` | `rade_initialize()` / `rade_finalize()` (no-ops) |
| `struct rade *rade_open(char model_file[], int flags)` | `Rade? rade_open(string model_file, int flags)` |
| `void rade_close(struct rade *)` | `rade_close(Rade)` (the handle is unusable afterwards) |
| `int rade_version(void)` / `rade_version_minor` | same (returns 2 / 2) |
| `rade_n_tx_out`, `rade_n_tx_eoo_out`, `rade_nin_max`, `rade_nin`, `rade_n_features_in_out`, `rade_n_eoo_bits` | same |
| `int rade_tx(r, RADE_COMP tx_out[], float features_in[])` | `int rade_tx(Rade r, Span<RadeComp> tx_out, ReadOnlySpan<float> features_in)` |
| `void rade_tx_set_eoo_bits(r, float eoo_bits[])` | `rade_tx_set_eoo_bits(Rade, ReadOnlySpan<float>)` |
| `int rade_tx_eoo(r, RADE_COMP tx_eoo_out[])` | `int rade_tx_eoo(Rade, Span<RadeComp>)` |
| `int rade_rx(r, float features_out[], int *has_eoo_out, float eoo_out[], RADE_COMP rx_in[])` | `int rade_rx(Rade, Span<float> features_out, out int has_eoo_out, Span<float> eoo_out, ReadOnlySpan<RadeComp> rx_in)`. Pass an empty `eoo_out` for V2. |
| `rade_sync`, `rade_freq_offset`, `rade_snrdB_3k_est` | same |
| `void rade_get_stats(r, struct rade_stats *)` | `rade_get_stats(Rade, out RadeStats)` |
| `rade_set_disable_unsync`, `rade_tx_set_data_symbol`, `rade_rx_get_data_symbol`, `rade_rx_set_agc` | same |

Diagnostics the C library prints to stderr (the `rade_open` banner, per-frame
receiver status when verbose) go to `RadeLog.Writer`. It defaults to
`Console.Error`; set it to `null` to silence them.

Differences you may notice:

- Buffers are spans. Sizes are checked by the runtime, not trusted.
- Several contexts can coexist. Each context has its own `rand()` state, whereas C
  shares one process-global generator.
- All methods are allocation-free on the audio path. A context is not
  thread-safe; use one per stream.

## `RadeModem`: idiomatic wrapper

```csharp
using var m = new RadeModem(RadeMode.V2);          // txBandpass, verbose optional
int n = m.Transmit(features, iq);                  // FeaturesPerFrame floats -> TxSamplesPerFrame IQ
m.TransmitEndOfOver(iq);
var r = m.Receive(iq.AsSpan(0, m.RxSamplesNeeded), features);   // r.HasFeatures, r.EndOfOver
bool sync = m.InSync; float snr = m.SnrDb3k; RadeStats s = m.Stats;
```

## Vocoder

- `LpcnetEncoder.ComputeFeatures(short[160] pcm, float[36] features)`: speech to features.
- `Fargan.Cont` / `Fargan.Synthesize`: the raw `fargan_*` calls.
- `FarganStream.Push(float[36], short[160])`: `lpcnet_demo -fargan-synthesis`
  semantics. The first 5 frames prime the vocoder.

## Text and voice (`RadeSharp.Text`)

- `RadeText`: `rade_text_create`, `rade_text_generate_tx_string`, `rade_text_rx`,
  `rade_text_set_rx_callback`, `rade_text_enable_stats_output`. Same names as the
  freedv-gui C code, plus the `GenerateTxString` / `Rx` instance methods.
- `RadeVoiceModem`: speech ↔ IQ with the callsign in the V1 EOO frame. It is the
  managed equivalent of `zeus_rade.h`:

| `zeus_rade.h` | `RadeVoiceModem` |
|---|---|
| `zeus_rade_open` / `_close` | `new RadeVoiceModem()` / `Dispose()` |
| `zeus_rade_nin` / `_nin_max` / `_max_pcm_per_rx` | `RxSamplesNeeded` / `RxMaxSamples` / `MaxPcmPerReceive` |
| `zeus_rade_rx(z, iq, pcm)` | `Receive(iq, pcm)` |
| `zeus_rade_n_speech_samples` / `_n_tx_out` / `_n_tx_eoo_out` | `SpeechSamplesPerTx` / `TxSamplesPerFrame` / `TxEooSamples` |
| `zeus_rade_tx` / `_tx_eoo` | `Transmit` / `TransmitEndOfOver` |
| `zeus_rade_set_tx_callsign` | `SetTxCallsign` |
| `zeus_rade_sync` / `_freq_offset` / `_snr_db` | `InSync` / `FrequencyOffsetHz` / `SnrDb` |
| `zeus_rade_get_eoo_callsign` | `TakeEooCallsign()` (returns `null` when none) |
