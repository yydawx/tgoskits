cmake_minimum_required(VERSION 3.16)
project(ort_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ORT_BUILD /workspace/onnxruntime/build_riscv64)

add_executable(test_inference test_inference.cc)

target_compile_options(test_inference PRIVATE -Wno-unused-result)

target_include_directories(test_inference PRIVATE
    /workspace/onnxruntime/include/onnxruntime/core/session
    /workspace/onnxruntime/include/onnxruntime
    /workspace/onnxruntime/onnxruntime/core/session
)

file(GLOB ABSL_LIBS "${ORT_BUILD}/_deps/abseil_cpp-build/absl/*/*.a")

set(ORT_LIBS
    ${ORT_BUILD}/libonnxruntime_session.a
    ${ORT_BUILD}/libonnxruntime_framework.a
    ${ORT_BUILD}/libonnxruntime_providers.a
    ${ORT_BUILD}/libonnxruntime_graph.a
    ${ORT_BUILD}/libonnxruntime_common.a
    ${ORT_BUILD}/libonnxruntime_mlas.a
    ${ORT_BUILD}/libonnxruntime_util.a
    ${ORT_BUILD}/libonnxruntime_flatbuffers.a
    ${ORT_BUILD}/libonnxruntime_optimizer.a
    ${ORT_BUILD}/libonnxruntime_lora.a
    ${ORT_BUILD}/libonnx.a
    ${ORT_BUILD}/libonnx_proto.a
    ${ORT_BUILD}/_deps/flatbuffers-build/libflatbuffers.a
    /workspace/protobuf_riscv64/libprotobuf-lite.a
    /workspace/re2_manual/libre2.a
)

target_link_libraries(test_inference PRIVATE
    -Wl,--start-group ${ORT_LIBS} -Wl,--end-group
    -Wl,--start-group ${ABSL_LIBS} -Wl,--end-group
    pthread rt dl
)

set_target_properties(test_inference PROPERTIES LINK_FLAGS "-static")
