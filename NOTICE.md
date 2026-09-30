# Notices

RadeSharp is a derivative work (a port to C#) of the projects below. Their
copyright notices and licenses apply to the corresponding parts of this code.

## rade_c: BSD-2-Clause
Copyright (c) 2026 Peter B Marks (LICENSE); source files (C) 2024-2025 David Rowe, parts (C) 2022 Amazon, written by Jan Buethe.
https://github.com/freedv/rade_c. Ported in `src/RadeSharp` (API, DSP, V1, V2,
Core) and the RADE weights in `src/RadeSharp/Weights/rade_*.bin`.

## Opus: BSD-3-Clause
Copyright (c) Xiph.Org Foundation, Skype Limited, Octasic, Jean-Marc Valin,
Timothy B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding, Erik de Castro Lopo,
Mozilla, Amazon. https://github.com/xiph/opus. Ported in `src/RadeSharp/Nnet`,
`src/RadeSharp/Vocoder`, and `fargan.bin`, `pitchdnn.bin`, `lpcnet_tables.bin`.

## freedv-gui rade_text.c: BSD-2-Clause
Copyright (C) 2024 Mooneer Salem. Ported in `src/RadeSharp.Text/RadeText.cs`.

## codec2: LGPL-2.1
Copyright (C) David Rowe and contributors; Demod2D/Somap derived from the CML
library, Copyright (C) 2006 Matthew C. Valenti. https://github.com/drowe67/codec2
(v1.2.0). Ported in `src/RadeSharp.Text/Ldpc.cs`, `Phi0.cs` and `GpInterleaver.cs`.
Because of this code, the `RadeSharp.Text` assembly is distributed under
LGPL-2.1-or-later. It is a separate assembly so it can be replaced.

## Zeus zeus_rade shim: BSD-2-Clause
`src/RadeSharp.Text/RadeVoiceModem.cs` follows the behaviour of Zeus'
`native/radae/shim/zeus_rade.c`.
