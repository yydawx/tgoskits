"""
Run the same Conv models on x86 with the same deterministic input as the RISC-V test,
then compare hex outputs.
"""
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import struct

# Same input as ort_conv_test.c: (i % 17 - 8) * 0.1
def make_input(n):
    return np.array([(i % 17 - 8) * 0.1 for i in range(n)], dtype=np.float32)

def hex_str(val):
    bits = struct.unpack('!I', struct.pack('!f', val))[0]
    return f"0x{bits:08x}"

def run_model(onnx_path, input_shape):
    data = make_input(np.prod(input_shape)).reshape(input_shape)
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_opt)
    inp = {sess.get_inputs()[0].name: data}
    out = sess.run(None, inp)[0]
    return out.flatten()

# RISC-V outputs from QEMU run
riscv_conv1x1 = [
    0x3d23d709, 0x3d8f5c29, 0x3dcccccd, 0x3e051eb8, 0x3e23d70a, 0x3e428f5c,
    0x3e6147ae, 0x3e800000, 0xbd75c292, 0xbecccccd, 0xbea8f5c3, 0xbe851eb8,
    0xbe428f5c, 0xbdf5c290, 0xbd4ccccc, 0x3ca3d708, 0x3db851ee, 0xbf051eb9,
    0xbd23d708, 0x3d8f5c2c, 0x3e3851ec, 0x3e947ae2, 0x3eccccce, 0x3f028f5d,
    0x3f1eb852, 0x3f3ae148, 0xbe3851ec
]

riscv_conv3x3 = [
    0x3f0f5c28, 0x3f8147ae, 0x3fbae148, 0x3fa3d70a, 0x3ebd70a2, 0xbebd70a4,
    0xbd23d720, 0xbee147ae, 0xbf2b8520
]

riscv_conv_bn = [
    0xbebd7066, 0xbe147ab0, 0x3da3d6d4, 0xbc23d6e0, 0xbee147ae, 0xbf55c24a,
    0xbf2b84e6, 0xbf5eb80a, 0xbf7c28a4, 0xbf45ce92, 0x3e915514, 0x3fab91d4,
    0x3ff8d8fd, 0x3f13cc2a, 0xbf259b98, 0x3ec4d9d9, 0xbf0dfeb8, 0xbacf62a
]

ONNX_DIR = "/home/yyda/workspace/tgoskits/test_ort/conv_models"

def compare(name, onnx_path, input_shape, riscv_hex):
    out = run_model(onnx_path, input_shape)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    all_match = True
    for i in range(len(out)):
        x86_hex = hex_str(out[i])
        rv_hex = f"0x{riscv_hex[i]:08x}"
        x86_val = out[i]
        # Decode RISC-V hex to float
        rv_bits = riscv_hex[i]
        rv_val = struct.unpack('!f', struct.pack('!I', rv_bits))[0]
        diff = abs(x86_val - rv_val)
        match = x86_hex == rv_hex
        if not match:
            all_match = False
        status = "OK" if match else "MISMATCH"
        print(f"  [{i:2d}] x86={x86_hex} ({x86_val:+.8f})  rv={rv_hex} ({rv_val:+.8f})  diff={diff:.2e}  {status}")

    print(f"\n  Result: {'ALL MATCH' if all_match else 'HAS MISMATCHES'}")
    return all_match

compare("conv1x1 (1x1 Conv, GEMM only)",
        f"{ONNX_DIR}/conv1x1.onnx", (1,2,3,3), riscv_conv1x1)

compare("conv3x3 (3x3 Conv, im2col+GEMM)",
        f"{ONNX_DIR}/conv3x3.onnx", (1,1,5,5), riscv_conv3x3)

compare("conv_bn (3x3 Conv + BatchNorm)",
        f"{ONNX_DIR}/conv_bn.onnx", (1,1,5,5), riscv_conv_bn)
