# Porting ONNX Runtime to StarryOS — Decision Log

## Context

赛题要求在 StarryOS 上完成 ACT 模型推理。核心任务是**移植推理框架到自研 OS**，而非仅仅调用现成库。

tract-onnx（纯 Rust）已在 QEMU RISC-V 上跑通，但：
- 移植成本为零（不体现 OS 能力）
- 性能较差（33.7s/次 QEMU 软模拟）
- 无法利用硬件加速（SIMD/RKNN delegate）

因此决策：**移植 Microsoft ONNX Runtime 到 StarryOS**。

## D-001: 选择 onnxruntime 而非其他框架

| 方案 | 理由 |
|------|------|
| **onnxruntime** ✅ | C/C++，官方支持 Linux/RISC-V，可裁剪（`--minimal`），赛题官方提及 |
| TVM | 需 LLVM/Python 工具链，交叉编译复杂 |
| ncnn | 腾讯出品，轻量但 ONNX 支持不完整 |
| MNN | 阿里出品，侧重移动端，RISC-V 支持不确定 |

## D-002: 构建策略 — Minimal Build

onnxruntime 完整构建依赖 protobuf/eigen/abseil 等大型依赖。选择 `--minimal` 构建：

- 禁用 Python/Training/Tests/Shared Libs
- 内置 ONNX 解析器（替代 protobuf）
- 最小化 syscall 依赖面

关键 cmake 选项：
```
-Donnxruntime_BUILD_SHARED_LIB=OFF
-Donnxruntime_DISABLE_CONTRIB_OPS=ON
-Donnxruntime_DISABLE_ML_OPS=ON
-Donnxruntime_MINIMAL_BUILD=ON
-Donnxruntime_DISABLE_FLOAT16_OPS=ON
-DCMAKE_BUILD_TYPE=Release
```

## D-003: 静态链接 musl

StarryOS 用户态基于 musl libc。交叉编译目标：`riscv64gc-unknown-linux-musl`。

- 静态链接避免动态链接器问题
- musl 的 syscall 层与 StarryOS 的 POSIX 兼容层对接
- 可能需要补全 StarryOS 中缺失的 syscall

## D-004: 跨架构验证流程

```
x86_64 minimal build (验证功能正确)
  → riscv64 cross-compile (验证编译通过)
    → StarryOS QEMU (验证 syscall 兼容)
      → 实机 RK3588/SG2002 (验证性能)
```

## D-005: 性能优化路线

- **短期**：确保 onnxruntime 在 QEMU 上能跑通
- **中期**：利用 RISC-V V 扩展（向量指令）加速矩阵运算
- **长期**：SG2002 上用 RKNN delegate 卸载计算到 NPU
