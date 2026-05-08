"""
Create minimal Conv test models for RISC-V vs x86 comparison.
"""
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import struct, os

OUT_DIR = "/home/yyda/workspace/tgoskits/test_ort/conv_models"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Model 1: 1x1 Conv (tests GEMM only, no im2col) ───
class Conv1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=1, bias=True)
        # deterministic weights
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(1, 7, dtype=torch.float32).reshape(3, 2, 1, 1) * 0.1)
            self.conv.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
    def forward(self, x):
        return self.conv(x)

# ─── Model 2: 3x3 Conv (tests im2col + GEMM) ───
class Conv3x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 3, 3) * 0.1)
            self.conv.bias.copy_(torch.tensor([0.5], dtype=torch.float32))
    def forward(self, x):
        return self.conv(x)

# ─── Model 3: Conv3x3 + BatchNorm ───
class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(2, eps=1e-5, momentum=0.1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(1, 19, dtype=torch.float32).reshape(2, 1, 3, 3) * 0.05)
            self.conv.bias.copy_(torch.tensor([0.1, -0.1], dtype=torch.float32))
            # BN stats
            self.bn.running_mean.copy_(torch.tensor([0.5, -0.3]))
            self.bn.running_var.copy_(torch.tensor([1.0, 0.8]))
            self.bn.weight.copy_(torch.tensor([1.0, 1.5]))  # gamma
            self.bn.bias.copy_(torch.tensor([0.0, 0.2]))     # beta
    def forward(self, x):
        return self.bn(self.conv(x))

def export_and_verify(name, model, input_shape):
    onnx_path = os.path.join(OUT_DIR, f"{name}.onnx")
    ort_path  = os.path.join(OUT_DIR, f"{name}.ort")
    hex_path  = os.path.join(OUT_DIR, f"{name}_x86_output.txt")

    dummy = torch.randn(*input_shape)
    model.eval()

    # Export ONNX
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=18, do_constant_folding=True)
    print(f"[{name}] ONNX exported: {onnx_path}")

    # Simplify (no-op for these tiny models, but keeps pipeline consistent)
    # Skip onnxsim for minimal models

    # Run x86 inference with ORT
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_opt)
    inp = {sess.get_inputs()[0].name: dummy.numpy()}
    out = sess.run(None, inp)[0]
    print(f"[{name}] x86 output shape: {out.shape}")
    print(f"[{name}] x86 output: {out.flatten().tolist()}")

    # Save x86 output as hex for comparison
    with open(hex_path, "w") as f:
        f.write(f"shape: {list(out.shape)}\n")
        f.write(f"data (hex):\n")
        for val in out.flatten():
            bits = struct.unpack('!I', struct.pack('!f', val))[0]
            f.write(f"  0x{bits:08x}\n")
        f.write(f"data (float):\n")
        for val in out.flatten():
            f.write(f"  {val:.8f}\n")
    print(f"[{name}] x86 output saved: {hex_path}")

    # Convert to ORT
    import onnxruntime as ort_convert
    so = ort_convert.SessionOptions()
    so.optimized_model_filepath = ort_path
    so.graph_optimization_level = ort_convert.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_convert.InferenceSession(onnx_path, so)
    print(f"[{name}] ORT exported: {ort_path}")

    return out

# Export all
np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("Model 1: 1x1 Conv (GEMM only)")
print("=" * 60)
out1 = export_and_verify("conv1x1", Conv1x1(), (1, 2, 3, 3))

print()
print("=" * 60)
print("Model 2: 3x3 Conv (im2col + GEMM)")
print("=" * 60)
out2 = export_and_verify("conv3x3", Conv3x3(), (1, 1, 5, 5))

print()
print("=" * 60)
print("Model 3: 3x3 Conv + BatchNorm")
print("=" * 60)
out3 = export_and_verify("conv_bn", ConvBN(), (1, 1, 5, 5))

print()
print("All models created in:", OUT_DIR)
