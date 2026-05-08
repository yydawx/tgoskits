---
name: milestone-summary
description: 阶段性总结：更新 git 状态（过滤构建产物）、提交代码、更新进度文档。当用户说"阶段性总结"、"收尾"、"总结一下进度"时使用此技能。
---

# 阶段性总结

## 概述

本技能用于在开发里程碑节点进行代码提交和进度文档更新。核心原则：**提交源码、配置和体现开发过程的脚本/测试程序，不提交编译产物、二进制和模型文件。**

## 工作流程

### 1. 检查 git 状态

```bash
git status
git log --oneline -5
```

分析所有未跟踪和已修改的文件，分为三类：

**可提交**：
- 源码：`*.rs`, `*.c`, `*.py`
- 配置：`*.toml`（Cargo.toml、qemu-riscv64.toml 等）
- `.gitignore`
- 构建/测试脚本：`*.sh`（如 `build_conv_test.sh`、`create_conv_models.py`，体现开发过程）
- 测试源码：`ort_test.c`、`ort_conv_test.c`、`float_diag.c` 等（C 测试程序源码，即使在根目录）
- 项目文档：`CLAUDE.md`、`doc/` 下文档

**不提交**（构建产物）：
- 编译后的 ELF 二进制：`ort-hello`、`ort-conv-test`、`float-diag` 等无后缀可执行文件
- 模型文件：`*.ort`、`*.onnx`、`*.pt`、`*.bin`
- 编译输出目录：`onnxruntime/build_riscv64/`、`ort_install/`、`target/`
- 第三方依赖：`protobuf_riscv64/`、`re2_riscv64/`、`re2_manual/`
- 工具链文件：`riscv64-*-toolchain.cmake`
- 外部数据：`AKA-Sim2Real/`

**不提交但保留**（进度文档）：
- `doc/ort-progress.md` — 只更新不提交

### 2. 更新 .gitignore

发现新的构建产物不在 `.gitignore` 中则添加。格式：
```
# 注释说明
路径或通配符
```

### 3. 暂存并提交

只暂存"可提交"的文件，**不用** `git add .` 或 `git add -A`。

```bash
git add <具体文件1> <具体文件2> ...
```

提交信息格式：`type(scope): 内容`
- `feat(starry)` — 新功能
- `fix(starry)` — 修复
- `docs` — 文档
- `chore` — 杂项

共同作者：`Co-Authored-By: xiaomi mimo <noreply@xiaomi.com>`

### 4. 更新进度文档

更新 `doc/ort-progress.md`（不提交），包含：
- **完成项**：新增已完成工作
- **待解决问题**：当前问题及状态
- **文件说明**：新增文件用途

### 5. 展示进度

```
## 本次提交
- hash: xxxxxxx
- 变更: ...
- 信息: ...

## 当前进度
- [x] 已完成
- [ ] 待完成

## 待解决问题
1. 问题 — 状态
```

## 注意事项

- 不提交编译后的 ELF、`.ort`、`.onnx`、`.bin` 等二进制/模型文件
- C 源码（`*.c`）、Python 脚本（`*.py`）、Shell 脚本（`*.sh`）体现开发过程，**应该提交**
- 进度文档只更新不提交
- 用户明确指定的文件以用户为准
- Co-Authored-By 统一写 `xiaomi mimo`
