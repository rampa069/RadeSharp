#!/usr/bin/env bash
# Rebuilds everything derived from upstream C after bumping reference/pins.cmake
# (or checking out a new rade_c): scalar reference build, weight blobs, model
# bindings, phi0 table, golden vectors; then runs the parity test suite.
#
#   scripts/regenerate.sh              (paths default to the sibling layout)
#   RADE_C_DIR=... FREEDV_TEXT_DIR=... BUILD_DIR=... scripts/regenerate.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
PARENT="$(dirname "$REPO")"
RADE_C_DIR="${RADE_C_DIR:-$PARENT/rade_c}"
FREEDV_TEXT_DIR="${FREEDV_TEXT_DIR:-$PARENT/../zeus/native/radae/vendor/freedv_text}"
BUILD_DIR="${BUILD_DIR:-$PARENT/rade_ref-build}"   # must not contain '#'

want="$(sed -n 's/^set(RADE_C_SHA "\([0-9a-f]*\)").*/\1/p' "$REPO/reference/pins.cmake")"
have="$(git -C "$RADE_C_DIR" rev-parse HEAD)"
[ "$want" = "$have" ] || { echo "rade_c checkout is $have but pins.cmake says $want" >&2; exit 1; }

cmake -G Ninja -S "$REPO/reference" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DRADE_C_DIR="$RADE_C_DIR" -DFREEDV_TEXT_DIR="$FREEDV_TEXT_DIR"
cmake --build "$BUILD_DIR"

OPUS_DNN="$BUILD_DIR/build_opus-prefix/src/build_opus/dnn"
python3 "$HERE/gen_models.py" "$REPO/src/RadeSharp/Models/Generated" \
  rade_c/src/rade_enc_data.c="$RADE_C_DIR/src/rade_enc_data.c":RadeEncModel \
  rade_c/src/rade_dec_data.c="$RADE_C_DIR/src/rade_dec_data.c":RadeDecModel \
  rade_c/src/rade_enc_v2_data.c="$RADE_C_DIR/src/rade_enc_v2_data.c":RadeEncV2Model \
  rade_c/src/rade_dec_v2_data.c="$RADE_C_DIR/src/rade_dec_v2_data.c":RadeDecV2Model \
  rade_c/src/rade_sync_data.c="$RADE_C_DIR/src/rade_sync_data.c":RadeSyncModel \
  opus/dnn/fargan_data.c="$OPUS_DNN/fargan_data.c":FarganModel \
  opus/dnn/pitchdnn_data.c="$OPUS_DNN/pitchdnn_data.c":PitchDnnModel
python3 "$HERE/gen_phi0.py" "$FREEDV_TEXT_DIR/codec2/phi0.c" "$REPO/src/RadeSharp.Text/Phi0.cs"

"$REPO/reference/make_golden.sh" "$BUILD_DIR"

cd "$REPO"
dotnet test RadeSharp.slnx
git status --short
