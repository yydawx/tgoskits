# Technical Findings

## F-001: StarryOS 已验证的能力

| 能力 | 验证方式 | 状态 |
|------|---------|------|
| Rust std 程序运行 | starry-hello | ✅ |
| 静态链接 ELF 加载 | tract-hello (17MB) | ✅ |
| 大文件读取 (194MB) | act_model.onnx | ✅ |
| mmap 基础功能 | tract-onnx 推理 | ✅ |
| 文件 I/O | std::fs::read | ✅ |
| 时钟 (clock_gettime) | std::time::Instant | ✅ |

## F-002: tract-onnx 的 syscall 清单

通过 `strace` 和代码分析，tract-onnx 在 Linux 上使用的 syscall：
- `mmap`, `munmap`, `mprotect` — 内存管理
- `read`, `write`, `open`, `close`, `stat` — 文件 I/O
- `clock_gettime` — 计时
- `brk` — 堆扩展
- `exit_group` — 进程退出

**无 futex/clone 调用**（单线程纯计算），这也是 tract-onnx 能零移植的原因。

## F-003: onnxruntime 的 syscall 依赖预估

onnxruntime 完整版比 tract-onnx 多出的 syscall：

| Syscall | 用途 | StarryOS 状态 |
|---------|------|--------------|
| `mmap` | 内存分配 | ✅ 已支持 |
| `mprotect` | 内存保护 | ✅ 已支持 |
| `futex` | 线程同步 | ⚠️ 需确认 |
| `clone` / `clone3` | 线程创建 | ⚠️ 需确认 |
| `sched_getaffinity` | CPU 亲和性 | ⚠️ 需确认 |
| `getrandom` | 随机数 | ⚠️ 需确认 |
| `pipe2` | 管道 | ⚠️ 需确认 |
| `sigaltstack` | 信号栈 | ⚠️ 需确认 |
| `rseq` | restartable sequences | ❌ 大概率缺失 |

**Minimal build 可能减少到仅需 mmap/mprotect/read/write/brk/clock_gettime**，与 tract-onnx 相当。

## F-004: 真实 ACT 模型特征

| 属性 | 值 |
|------|-----|
| 参数量 | ~22M |
| ONNX 文件大小 | 193.5 MB (merged) |
| ONNX 算子数 | 748 nodes |
| 输入 | image (1,3,224,224) + state (1,2) |
| 输出 | action (1,8,2) = 16 f32 |
| 关键算子 | Conv, BatchNorm, MatMul, GELU, LayerNorm, Softmax, Transpose |
| tract-onnx 兼容 | ✅ 全部支持 |

## F-005: QEMU 内存需求

| 组件 | 内存占用 |
|------|---------|
| StarryOS 内核 | ~50MB |
| 用户态二进制 (tract-onnx) | ~17MB |
| ONNX 模型 (194MB 文件) | ~250MB (加载后) |
| 运行时开销 | ~50MB |
| **总计** | **~370MB** |
| QEMU 分配 | 1024MB ✅ |

512MB 对真实 ACT 模型不够（模型加载时峰值超 512MB），需 1024MB。

## F-006: onnxruntime RISC-V 官方支持

onnxruntime 从 v1.16 开始有 RISC-V 交叉编译支持：
- CI 有 riscv64 Linux 构建
- 使用 gcc 交叉工具链
- 默认启用 pthread
- 需要 `-Donnxruntime_BUILD_SHARED_LIB=OFF` 静态链接
