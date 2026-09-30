# Upstream versions this port tracks. Bump together with the C# port (see UPDATING.md).
set(RADE_C_SHA "cc17222acc597339199bdfd7253c3e6cf147f953")  # freedv/rade_c
set(OPUS_SHA   "940d4e5af64351ca8ba8390df3f555484c567fbb")  # xiph/opus (dnn/, FARGAN, LPCNet)
# FreeDV reliable text (EOO callsign): freedv-gui src/pipeline/rade_text.c + codec2 v1.2.0 LDPC slice,
# as vendored by Zeus (native/radae/vendor/freedv_text) from sv1eia/Thetis-RADE at this SHA.
set(FREEDV_TEXT_SHA "f7605a46bd21275ab8b9edd00d4a1b6fae6eabe8")
