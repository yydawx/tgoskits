# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概览

TGOSKits 是清华大学 RCore 团队的 OS 与虚拟化一体化开发仓库。通过 Git Subtree 将 60+ 个独立组件仓库整合到一个 Cargo workspace 中，基于共享组件构建三套目标系统：

- **ArceOS** (`os/arceos/`): 模块化 unikernel 操作系统
- **StarryOS** (`os/StarryOS/`): 面向教学的 POSIX 兼容操作系统
- **Axvisor** (`os/axvisor/`): 基于 ArceOS 的虚拟机监控器

## 当前竞赛目标

本仓库当前承载 **2026 年全国大学生计算机系统能力大赛 - OS 功能挑战赛道** 的赛题开发。

**赛题**：面向具身智能的国产操作系统 ACT 模型适配与实时推理优化（Porting and Real-time Inference Optimization of Action Chunking Transformer on StarryOS for Embodied Intelligence）

**赛题核心任务**：在 StarryOS 上完成 ACT（Action Chunking Transformer）模型推理，输出预期数据。ACT 是一种模仿学习模型，用于双臂精细操作（论文：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*）。

### 三个子任务

| 子任务 | 目标平台 | 评审等级 |
|--------|----------|----------|
| 任务一 | 荔枝派 SG2002（RISC-V，256MB 内存） | 一等奖（最高难度） |
| 任务二 | 香橙派 RK3588（ARM，4GB 内存） | 二等奖 |
| 任务三 | QEMU 模拟环境 | 三等奖（基础达标） |

**评审等级**：
- 三等奖：完成任务三
- 二等奖：完成任务三 + 任务二
- 一等奖：完成任务一
- 特等奖：同时完成任务一和任务二，且工程实现与优化表现良好

### 关键约束与技术要点

- 嵌入式平台资源受限（内存、算力），需要模型裁剪/量化、推理框架适配（如 ONNX Runtime）、性能优化（启动时间、推理延迟）
- 软硬件协同，涉及操作系统、驱动、AI 推理部署多方面能力
- RK3588 可使用 RKNN 模型转换工具：https://github.com/airockchip/rknn-toolkit2
- 荔枝派参考：https://wiki.sipeed.com/hardware/zh/lichee/RV_Nano/5_peripheral.html
- 赛题维护：北京亚嵌科技有限责任公司，联系 limingth@qq.com

**开发时应始终以赛题子任务的完成为导向，所有改动需服务于在目标平台上跑通 ACT 推理这一核心目标。**

## 构建与开发命令

所有构建和测试均通过 `cargo xtask` 完成。Cargo 别名定义在 `.cargo/config.toml`。

### 构建与运行（QEMU）

```sh
cargo xtask arceos qemu --package <pkg> --arch <arch>    # ArceOS
cargo xtask starry qemu --arch <arch>                     # StarryOS
cargo xtask axvisor qemu --arch <arch>                    # Axvisor
```

支持架构：`aarch64`、`riscv64`、`x86_64`、`loongarch64`。

### 测试

```sh
cargo xtask arceos test qemu --arch <arch>   # ArceOS QEMU 测试（test-suit/arceos/rust/）
cargo xtask starry test qemu --arch <arch>   # StarryOS QEMU 测试（test-suit/starryos/）
cargo xtask axvisor test qemu --arch <arch>  # Axvisor QEMU 测试
cargo xtask test                              # 全量回归（QEMU 测试 + std 测试）
```

- ArceOS 测试：`test-suit/arceos/rust/` 下的 Rust 包（分类：display, exception, fs, memtest, net, task）。
- StarryOS 测试：`test-suit/starryos/normal/` 和 `test-suit/starryos/stress/` 下按目录组织，每个用例包含 `qemu-<arch>.toml` 配置（shell 命令、success/fail 正则、超时时间）。

### Lint 与格式化

```sh
cargo fmt --all                                  # 格式化全部代码（修改后必须执行）
cargo xtask clippy --package <crate>             # 针对单个 crate 的 clippy 检查
cargo xtask clippy --all                         # 对白名单内所有 crate 执行 clippy
cargo xtask sync-lint                            # 原子操作排序检查（检测可疑的 Relaxed 使用）
```

Clippy 白名单在 `scripts/test/clippy_crates.csv`。若某 crate 通过 clippy 但不在白名单中，应将其加入。

Starry rootfs 构建：`cargo xtask starry rootfs --arch <arch>`

## 代码架构

### 工作区布局

- `components/` — 50+ 可复用组件 crate（allocator、CPU、调度、驱动、虚拟化组件、文件系统、网络等），通过 `scripts/repo/repo.py` 从上游独立仓库以 Git Subtree 形式同步。
- `os/arceos/modules/` — ArceOS 内核模块（axalloc、axhal、axdriver、axfs、axmm、axnet、axtask 等）
- `os/arceos/api/` — API 层 crate：`axfeat`（feature flags）、`arceos_api`、`arceos_posix_api`
- `os/arceos/ulib/` — 用户库：`axstd`（bare-metal std）、`axlibc`（libc shim）
- `os/StarryOS/` — StarryOS 内核、配置与构建辅助
- `os/axvisor/` — Hypervisor 源码
- `platform/` 和 `components/axplat_crates/platforms/` — 各目标板的平台抽象 crate
- `scripts/axbuild/` — 核心构建库（`axbuild` crate），包含所有 xtask 逻辑
- `xtask/` — 薄入口，委托给 `scripts/axbuild/`
- `scripts/test/` — CSV 白名单：`clippy_crates.csv`、`std_crates.csv`
- `scripts/repo/` — Subtree 管理（`repo.py pull/push/list`）

### 构建系统内部

`xtask/src/main.rs` 是薄壳，所有逻辑在 `scripts/axbuild/` 中，模块包括：`arceos/`、`starry/`、`axvisor/`、`board.rs`、`clippy.rs`、`sync_lint.rs`、`test_qemu.rs`、`test_std.rs`、`rootfs/`。

所有工作区 crate 通过根 `Cargo.toml` 的 `[patch.crates-io]` 以 path 依赖替换。工作区使用 edition 2024、resolver 3、nightly-2026-04-01。

### Subtree 管理

`scripts/repo/repos.csv` 映射上游仓库到本地目录。使用 `python3 scripts/repo/repo.py list/pull/push` 管理 subtree。

## 开发规范

- 修改代码后必须执行 `cargo fmt`（使用 style_edition 2024，`group_imports = "StdExternalCrate"`，`imports_granularity = "Crate"`）。
- 修改逻辑后执行 `cargo clippy --package <crate>`。不要用 `#[allow]` 压制警告，除非用户明确要求。
- 优先使用 `cargo xtask`，不要直接用 `cargo build/test/run`。仅在 xtask 无法满足需求时才回退到原生 Cargo。
- PR 标题遵循 Conventional Commits 格式：`type(scope): content`。scope 用主要受影响的 crate 名（如 `feat(axbuild): add Starry remote board test flow`），跨模块改动可用 `ci`、`repo`、`docs` 等。
- 分支策略：feature 分支 -> dev（PR）-> main（常规合并）。禁止直接推送到 main。
