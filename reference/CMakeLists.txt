# Reference build of rade_c + Opus DNN used to generate golden vectors for RadeSharp.
#
# Built scalar (no SIMD) and without FMA contraction so the C results are the
# ones the managed port reproduces bit-for-bit. Configure with Ninja and a build
# directory OUTSIDE this repo: its path contains '#', which breaks Makefiles.
#
#   cmake -G Ninja -S reference -B ../rade_ref-build -DCMAKE_BUILD_TYPE=Release
#   cmake --build ../rade_ref-build

cmake_minimum_required(VERSION 3.16)
project(rade_reference C)
set(CMAKE_C_STANDARD 11)

set(RADE_C_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rade_c" CACHE PATH "rade_c checkout")
get_filename_component(RADE_C_DIR "${RADE_C_DIR}" ABSOLUTE)
set(RADE_SRC "${RADE_C_DIR}/src")
include("${CMAKE_CURRENT_SOURCE_DIR}/pins.cmake")

set(SCALAR_CFLAGS "-O2 -ffp-contract=off -DDISABLE_NEON -fno-fast-math")

include(ExternalProject)
ExternalProject_Add(build_opus
    URL https://github.com/xiph/opus/archive/${OPUS_SHA}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP NO
    BUILD_IN_SOURCE 1
    PATCH_COMMAND sh -c "patch dnn/nnet.h < '${RADE_SRC}/opus-nnet.h.diff' && patch dnn/nnet.c < '${RADE_SRC}/opus-nnet.c.diff'"
    CONFIGURE_COMMAND ./autogen.sh && ./configure --with-pic --enable-osce --enable-dred
        --disable-shared --disable-doc --disable-extra-programs
        --disable-intrinsics --disable-rtcd "CFLAGS=${SCALAR_CFLAGS}"
    BUILD_COMMAND make -j8
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS <SOURCE_DIR>/.libs/libopus.a
)
ExternalProject_Get_Property(build_opus SOURCE_DIR)
set(OPUS_DIR "${SOURCE_DIR}")
file(MAKE_DIRECTORY "${OPUS_DIR}/include")
add_library(opus STATIC IMPORTED)
add_dependencies(opus build_opus)
set_target_properties(opus PROPERTIES IMPORTED_LOCATION "${OPUS_DIR}/.libs/libopus.a")

separate_arguments(SCALAR_LIST UNIX_COMMAND "${SCALAR_CFLAGS}")
add_compile_options(${SCALAR_LIST})
include_directories("${RADE_SRC}" "${OPUS_DIR}/dnn" "${OPUS_DIR}/celt" "${OPUS_DIR}/include" "${OPUS_DIR}")

add_library(rade STATIC
    rade_api.c rade_enc.c rade_dec.c rade_enc_data.c rade_dec_data.c
    rade_dsp.c rade_ofdm.c rade_bpf.c rade_acq.c rade_tx.c rade_rx.c
    rade_enc_v2.c rade_dec_v2.c rade_sync.c rade_enc_v2_data.c rade_dec_v2_data.c rade_sync_data.c
    rade_v2_ofdm.c rade_tx_v2.c rade_rx_v2.c)
get_target_property(_srcs rade SOURCES)
list(TRANSFORM _srcs PREPEND "${RADE_SRC}/")
set_target_properties(rade PROPERTIES SOURCES "${_srcs}")
target_sources(rade PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tools/crand.c")
target_compile_definitions(rade PRIVATE IS_BUILDING_RADE_API=1)
set_source_files_properties(
    ${RADE_SRC}/rade_enc_data.c ${RADE_SRC}/rade_dec_data.c
    ${RADE_SRC}/rade_enc_v2_data.c ${RADE_SRC}/rade_dec_v2_data.c
    PROPERTIES COMPILE_OPTIONS "-O0")
target_link_libraries(rade PUBLIC opus m)

foreach(tool radae_tx radae_rx rade_tx_wav rade_rx_wav rade_v2_text_test
             rade_ber_test rade_v1_text_test rade_dec_v2_test rade_enc_v2_test)
    add_executable(${tool} "${RADE_SRC}/${tool}.c")
    target_link_libraries(${tool} rade)
endforeach()
add_executable(lpcnet_demo "${RADE_SRC}/lpcnet_demo.c")
target_link_libraries(lpcnet_demo opus m)
add_executable(real2iq "${RADE_SRC}/real2iq.c")
target_link_libraries(real2iq m)
add_executable(rade_bpf_test "${RADE_SRC}/rade_bpf_test.c" "${RADE_SRC}/kiss_fft.c" "${RADE_SRC}/kiss_fftr.c")
target_link_libraries(rade_bpf_test rade)

# ---- golden-vector / weight export tools (this repo) ----------------------
set(REF_TOOLS "${CMAKE_CURRENT_SOURCE_DIR}/tools")
add_executable(dump_weights "${REF_TOOLS}/dump_weights.c")
target_link_libraries(dump_weights rade)
add_executable(rade_trace "${REF_TOOLS}/rade_trace.c")
target_link_libraries(rade_trace rade)
add_executable(chan "${REF_TOOLS}/chan.c")
target_link_libraries(chan m)
add_executable(nn_golden "${REF_TOOLS}/nn_golden.c")
target_link_libraries(nn_golden rade)

# FreeDV reliable text (EOO callsign) -- the slice Zeus vendors from Thetis-RADE
set(FREEDV_TEXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../zeus/native/radae/vendor/freedv_text" CACHE PATH "freedv_text slice")
get_filename_component(FREEDV_TEXT_DIR "${FREEDV_TEXT_DIR}" ABSOLUTE)
add_executable(text_golden "${REF_TOOLS}/text_golden.c"
    "${FREEDV_TEXT_DIR}/src/rade_text.c"
    "${FREEDV_TEXT_DIR}/codec2/mpdecode_core.c" "${FREEDV_TEXT_DIR}/codec2/gp_interleaver.c"
    "${FREEDV_TEXT_DIR}/codec2/ldpc_codes.c" "${FREEDV_TEXT_DIR}/codec2/HRA_56_56.c" "${FREEDV_TEXT_DIR}/codec2/phi0.c")
target_include_directories(text_golden BEFORE PRIVATE "${FREEDV_TEXT_DIR}/src" "${FREEDV_TEXT_DIR}/codec2")
target_link_libraries(text_golden m)
