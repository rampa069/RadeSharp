# RadeSharp

A managed C# (.NET 10) port of [rade_c](https://github.com/freedv/rade_c), the
C implementation of FreeDV's Radio Autoencoder (RADE V1 and V2). It also ports the
pieces a radio application needs around it:

- the Opus FARGAN vocoder (features → 16 kHz speech),
- the LPCNet feature analysis (speech → features),
- the FreeDV reliable-text End-of-Over callsign codec.

It has no native dependencies. The weights are embedded, and it runs anywhere
.NET runs.

The port is **bit-exact** against a scalar, no-FMA build of the upstream C. The
test suite checks every stage against golden vectors produced by that build. The
WAV tools produce byte-identical files.

> ⚠️ RADE V2 is under active upstream development and not interoperable across
> versions. See the rade_c README.

## Layout

| Path | What |
|---|---|
| `src/RadeSharp` | RADE modem (V1/V2), Opus DNN core, FARGAN, LPCNet analysis (BSD) |
| `src/RadeSharp.Text` | FreeDV EOO text codec + `RadeVoiceModem` (LGPL-2.1, because of the codec2 LDPC code) |
| `tools/RadeSharp.Tools` | `radesharp-cli`: `radae_tx`, `radae_rx`, `lpcnet_demo`, `rade_tx_wav`, `rade_rx_wav` |
| `tools/RadeSharp.Bench` | real-time-factor benchmark |
| `tests/RadeSharp.Tests` | parity tests against `tests/fixtures` |
| `reference/` | scalar reference build of the C code and golden-vector generators |
| `scripts/` | regeneration and upstream-tracking tools |

## APIs

The library has three levels:

1. **`RadeApi`**: 1:1 with `rade_api.h`. Same function names, flags, constants and
   return values, with pointers replaced by spans. Code written against the C API
   ports mechanically. See [docs/API.md](docs/API.md).
2. **`RadeModem`**: an idiomatic `IDisposable` wrapper over one context, working
   on features ↔ 8 kHz IQ.
3. **`RadeVoiceModem`** (in `RadeSharp.Text`): speech ↔ IQ with FARGAN, LPCNet
   and the EOO callsign. It is the managed equivalent of Zeus' native `zeus_rade`
   shim.

```csharp
using RadeSharp;
using RadeSharp.Text;

using var modem = new RadeVoiceModem(RadeMode.V1);
modem.SetTxCallsign("EA5IUE");

// TX: 16 kHz speech -> 8 kHz IQ (transmit the real part as SSB audio)
var iq = new RadeComp[modem.TxSamplesPerFrame];
int n = modem.Transmit(speech.AsSpan(0, modem.SpeechSamplesPerTx), iq);

// RX: feed exactly RxSamplesNeeded IQ samples per call
var pcm = new short[modem.MaxPcmPerReceive];
int got = modem.Receive(rxIq.AsSpan(0, modem.RxSamplesNeeded), pcm);
string? call = modem.TakeEooCallsign();
```

## Build and test

```
dotnet build RadeSharp.slnx
dotnet test RadeSharp.slnx
dotnet run -c Release --project tools/RadeSharp.Bench
```

On an Apple M-series core, one stream uses a small fraction of real time: RX V1
while searching for sync ≈ 0.24×, FARGAN ≈ 0.12×, everything else < 0.04×.

## Reference build and updates

`reference/` builds rade_c + Opus scalar and without FMA contraction, plus tools
that dump weights and golden vectors. [UPDATING.md](UPDATING.md) is the procedure
for following new upstream versions. `scripts/upstream-diff.sh` maps upstream
changes to C# files, and `scripts/regenerate.sh` rebuilds the derived artifacts and
re-runs the parity suite.

## Licenses

See [NOTICE.md](NOTICE.md). The core (`RadeSharp`) is BSD-2-Clause (rade_c) /
BSD-3-Clause (Opus). `RadeSharp.Text` contains LGPL-2.1 code from codec2 and is a
separate assembly for that reason.
