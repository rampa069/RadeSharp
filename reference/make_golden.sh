#!/usr/bin/env bash
# Regenerates the golden vectors (tests/fixtures) and the weight blobs
# (src/RadeSharp/Weights) from the scalar reference build.
#
#   reference/make_golden.sh [build_dir]    (default: ../rade_ref-build next to the repo)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
B="${1:-$(dirname "$REPO")/rade_ref-build}"
RADE_C="${RADE_C_DIR:-$(dirname "$REPO")/rade_c}"
FX="$REPO/tests/fixtures"
W="$REPO/src/RadeSharp/Weights"
mkdir -p "$FX" "$W"

"$B/dump_weights" "$W"
"$B/nn_golden" "$FX"
"$B/text_golden" "$FX"

# 16 kHz s16 speech without the WAV header
python3 - "$RADE_C/input_sample.wav" "$FX/speech.s16" <<'PY'
import sys, wave
w = wave.open(sys.argv[1]); assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
open(sys.argv[2], "wb").write(w.readframes(w.getnframes()))
PY

"$B/lpcnet_demo" -features "$FX/speech.s16" "$FX/features.f32"
"$B/lpcnet_demo" -fargan-synthesis "$FX/features.f32" "$FX/fargan.s16"

for v in v1 v2; do
    flag=""; [ $v = v2 ] && flag="--v2"
    "$B/rade_trace" tx $flag "$FX/features.f32" "$FX/${v}_tx.iq"
    "$B/rade_trace" rx $flag "$FX/${v}_tx.iq" "$FX/${v}_rx.f32" "$FX/${v}_rx.trace"
    # impaired: gain 0.5, +23.5 Hz offset, noise, 1234 leading noise samples
    "$B/chan" 0.5 23.5 0.08 1234 7 < "$FX/${v}_tx.iq" > "$FX/${v}_ch.iq"
    "$B/rade_trace" rx $flag "$FX/${v}_ch.iq" "$FX/${v}_ch_rx.f32" "$FX/${v}_ch_rx.trace"
done

# WAV convenience tools end to end (speech -> modem audio -> speech)
for v in v1 v2; do
    flag=""; [ $v = v2 ] && flag="--v2"
    "$B/rade_tx_wav" $flag -v 0 "$RADE_C/input_sample.wav" "$FX/wav_tx_$v.wav"
    "$B/rade_rx_wav" $flag -v 0 "$FX/wav_tx_$v.wav" "$FX/wav_rx_$v.wav"
done

git -C "$RADE_C" rev-parse HEAD > "$FX/RADE_C_SHA"
ls -la "$FX" "$W"
