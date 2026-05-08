"""
Run ACT model on x86 with the same deterministic input as ort_test.c,
print output as hex for comparison with RISC-V.
"""
import onnxruntime as ort
import numpy as np
import struct

ORT_PATH = "/home/yyda/workspace/tgoskits/test-suit/starryos/normal/ort-inference/sh/act_model.ort"

sess_opt = ort.SessionOptions()
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(ORT_PATH, sess_opt)

print("=== ACT model info ===")
for inp in sess.get_inputs():
    shape = inp.shape
    print(f"  input '{inp.name}': type={inp.type} shape={shape}")
for out in sess.get_outputs():
    shape = out.shape
    print(f"  output '{out.name}': type={out.type} shape={shape}")

# Build deterministic inputs matching ort_test.c: (j % 17 - 8) * 0.1
feeds = {}
for inp in sess.get_inputs():
    name = inp.name
    shape = inp.shape
    # Resolve dynamic dims to 1
    resolved = []
    for d in shape:
        if isinstance(d, str) or d is None:
            resolved.append(1)
        else:
            resolved.append(d)
    n = 1
    for d in resolved:
        n *= d
    data = np.array([(j % 17 - 8) * 0.1 for j in range(n)], dtype=np.float32).reshape(resolved)
    feeds[name] = data
    print(f"  input '{name}': shape={resolved}, {n} elements")
    print(f"    first 8 hex: ", end="")
    for j in range(min(8, n)):
        bits = struct.unpack('!I', struct.pack('!f', data.flatten()[j]))[0]
        print(f"0x{bits:08x} ", end="")
    print()

print("\nRunning inference...")
outputs = sess.run(None, feeds)
out = outputs[0]
flat = out.flatten()
print(f"Output shape: {list(out.shape)} ({len(flat)} elements)\n")

print(f"output ({len(flat)} values):")
for i in range(len(flat)):
    bits = struct.unpack('!I', struct.pack('!f', flat[i]))[0]
    print(f"  [{i:3d}] 0x{bits:08x}")
