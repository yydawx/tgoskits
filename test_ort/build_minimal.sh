#!/bin/bash
cd /workspace

ABSL_LIBS=$(find /workspace/onnxruntime/build_riscv64/_deps/abseil_cpp-build -name '*.a' | tr '\n' ' ')

riscv64-linux-gnu-g++ -static -O2 -march=rv64gc -mabi=lp64d -Wno-error \
  minimal_ort_diag.c \
  -I/workspace/onnxruntime/include/onnxruntime/core/session \
  -I/workspace/onnxruntime/include \
  -Wl,--start-group \
  /workspace/onnxruntime/build_riscv64/combined/libonnxruntime_full.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_session.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_framework.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_common.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_mlas.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_lora.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_graph.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_optimizer.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_util.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_providers.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_providers_shared.a \
  /workspace/onnxruntime/build_riscv64/libonnxruntime_flatbuffers.a \
  /workspace/onnxruntime/build_riscv64/libonnx.a \
  /workspace/onnxruntime/build_riscv64/libonnx_proto.a \
  /workspace/onnxruntime/build_riscv64/_deps/re2-build/libre2.a \
  /workspace/onnxruntime/build_riscv64/_deps/protobuf-build/libprotobuf.a \
  /workspace/onnxruntime/build_riscv64/_deps/flatbuffers-build/libflatbuffers.a \
  /workspace/onnxruntime/build_riscv64/_deps/onnx-build/libonnx.a \
  /workspace/onnxruntime/build_riscv64/_deps/onnx-build/libonnx_proto.a \
  $ABSL_LIBS \
  -Wl,--end-group \
  -lpthread -ldl -lm \
  -o minimal-ort-diag 2>&1 | tail -5
echo "exit: $?"
