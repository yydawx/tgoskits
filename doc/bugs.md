# Known Issues & Workarounds

## B-001: tract-onnx 交叉编译 — 链接器找不到

**现象**: `cargo build --target riscv64gc-unknown-linux-musl` 报 `cc: command not found`

**原因**: 独立项目未继承 workspace 的 `.cargo/config.toml`

**解决**: 在项目 `.cargo/config.toml` 中配置：
```toml
[target.riscv64gc-unknown-linux-musl]
linker = "/path/to/riscv64-linux-musl-gcc"
```

或在 workspace 内构建（继承 workspace 配置）。

## B-002: PyTorch ONNX 导出 — 外部数据分离

**现象**: `torch.onnx.export` 生成 `.onnx` (几KB) + `.onnx.data` (几百MB)

**原因**: onnxscript-based exporter 默认使用外部数据格式

**解决**: 合并为单文件：
```python
import onnx
m = onnx.load("model.onnx", load_external_data=True)
onnx.save(m, "model_merged.onnx")
```

## B-003: PyTorch ONNX 导出 — Constant Folding 导致输出与输入无关

**现象**: 不同输入产生相同输出

**原因**: 固定尺寸的 input axes 导致 PyTorch 进行激进的常量折叠优化

**解决**: 使用 `dynamic_axes` 或不对输入做 shape 约束

## B-004: StarryOS 内存不足 — Kernel Panic

**现象**: `Unhandled Supervisor Page Fault` at 高地址

**原因**: phys-memory-size=128MB 不足以加载 17MB 二进制 + rootfs

**解决**: 在 `axconfig.toml` 中增大 `phys-memory-size`：
- 简单模型 (558KB): 512MB
- 真实 ACT (194MB): 1024MB

## B-005: Docker 容器中 docker 命令权限不足

**现象**: `docker cp` 报 `permission denied`

**解决**: 使用 `sg docker -c "docker ..."` 或检查用户 docker 组成员

## B-006: Rust 测试项目在 workspace 中编译失败

**现象**: `current package believes it's in a workspace when it's not`

**解决**: 在 Cargo.toml 中添加空 `[workspace]` 表，使其成为独立项目

## B-007: ONNX 模型输入维度错误 — 70 vs 14

**现象**: zeros/ones 测试 max_err=0.5，但前几个值完全匹配

**原因**: 参考值来自不同模型（70维输出），实际模型输出 14 维

**解决**: 用 tract-onnx 在 x86_64 上生成正确参考值，确保维度一致

## B-008: arr2 与 Vec<f32> 类型不匹配

**现象**: `mismatched types: expected array [_; _], found Vec<f32>`

**解决**: 使用 `Array2::from_shape_vec` 替代 `arr2`：
```rust
tract_ndarray::Array2::from_shape_vec((1, len), vec).unwrap().into_tensor()
```

## B-009: torch.onnx.export + torch 2.11 — dynamic_axes 导致 export 失败

**现象**: `GuardOnDataDependentSymNode: Could not guard on data-dependent expression`

**原因**: torch 2.11 的 dynamo-based exporter 不支持数据依赖分支

**解决**: 移除 forward 中的 `if tensor.sum() != 0` 等运行时条件判断，使用确定性推理路径

## B-010: torchvision 与 torch 版本不兼容

**现象**: `RuntimeError: operator torchvision::nms does not exist`

**原因**: torchvision 0.26.0 适配 torch 2.6.x，但系统装的是 torch 2.11.0

**解决**: 不依赖 torchvision，手写 ResNet-18 基础模块
