# ACT 模型 TPU 推理 — 完整开发历史

本文档记录在 SG2002（荔枝派 LicheeRV Nano）上完成 ACT 模型 TPU 推理的完整开发历程，包括所有踩坑、失败尝试和最终解决方案。

从 PyTorch 训练模型到板上 TPU 实时推理，历经 ONNX 导出、tract-onnx CPU 推理、ONNX Runtime 推理、ncnn CPU 量化推理、TPU-MLIR 编译、精度调试、内核编译、镜像拼装、批量验证等多个阶段。

**CPU 推理路线演进**：
```
tract-onnx (Gemm验证) → tract-onnx (完整ACT) → ONNX Runtime → ncnn INT8量化
     ✅ 12 PASS           ✅ QEMU 30min     ✅ QEMU ~15s    ✅ QEMU ~15s, 256MB PASS, 实测38s
```

最终 CPU 推理（ncnn INT8）产出了 38s 的板上推理时间，作为 TPU 加速的基线对比。

---

## 阶段〇：tract-onnx CPU 推理（2026-05-05 ~ 2026-05-25）

### 0.1 技术路线探索

在进入 ncnn 和 TPU 之前，首先尝试了 tract-onnx 作为 CPU 推理框架。

**选型考量**：
- tract 是纯 Rust 实现的 ONNX 推理引擎，可直接交叉编译到 RISC-V musl
- 无需 C/C++ 编译链，部署简单
- 适合在 StarryOS bare-metal 环境下运行

### 0.2 tract-onnx Gemm 验证（test-suit/starryos/normal/tract-inference）

**目标**：验证 tract-onnx 能否在 StarryOS RISC-V QEMU 上运行 ONNX 推理。

- 交叉编译 tract-onnx 到 `riscv64gc-unknown-linux-musl`
- 运行一个简单 Gemm 层推理
- 12 个测试用例（zeros、ones、10 个随机）全部通过与 ONNX Runtime x86 参考值对比

**关键文件**：
- `test-suit/starryos/normal/tract-inference/src/main.rs` — 推理程序（Gemm 模型）
- `test-suit/starryos/normal/tract-inference/build.sh` — 交叉编译脚本
- `test-suit/starryos/normal/tract-inference/qemu-riscv64.toml` — QEMU 测试配置

### 0.3 tract-onnx 完整 ACT 模型（test-suit/starryos/normal/act-inference）

**目标**：用 tract-onnx 在 QEMU RISC-V 上运行完整的 ACT 模型（194MB ONNX）。

- 输入：images (1,3,224,224) + state (1,2)，全零向量
- 成功加载并推理，但速度极慢（约 30 分钟）
- 在 QEMU 上验证了 ONNX 推理的可行性

**关键文件**：
- `test-suit/starryos/normal/act-inference/src/main.rs` — 完整 ACT ONNX 推理程序

**结论**：tract-onnx 可以工作，但性能不可接受。为后续 ONNX Runtime 和 ncnn 的尝试奠定了基础。

### 0.4 ONNX Runtime 交叉编译（test_ort/）

**目标**：用 ONNX Runtime (C API) 进行更快的 CPU 推理。

- 交叉编译 ONNX Runtime 到 RISC-V (`riscv64-linux-gnu-g++`)
- 编译 `ort_test.c` 推理程序
- QEMU 验证：FP16 模型 ~15s

**关键文件**：
- `build_act_test.sh`, `build_conv_test.sh`, `build_minimal.sh` — 交叉编译脚本
- `ort_test.c` — ORT 推理程序

**结论**：ORT 推理可用，速度接近 ncnn，但模型体积大（FP16 95MB），256MB 内存下 OOM。

---

## 阶段一：ONNX 导出与 ncnn 转换（2026-05-25）

### 1.1 PyTorch → ONNX

**目标**：将 PyTorch 训练的 ACT 模型导出为 ONNX 格式。

- 模型文件：`final_model.pt` (194MB FP32)
- 使用 `torch.onnx.export`，opset 14
- 修复 config 中 `temporal_ensembling_weight` → `temporal_ensembling_coeff` 键名不匹配

**产出**：`act_model.onnx` (194MB)

### 1.2 ONNX → ncnn (PNNX)

- `act_model.onnx` → PNNX → `act_model.ncnn.{param,bin}`
- FP16 模式：`pnnx fp16=1` → 95MB 模型
- FP32 模式：`pnnx fp16=0` → 190MB 模型

### 1.3 ruapu 崩溃修复

**问题**：ncnn 在 RISC-V 上的 `ruapu` CPU 特性探测通过执行非法指令来检测硬件能力。StarryOS 上 SIGILL 直接导致进程崩溃。

**修复**：Patch ncnn 源码 `src/cpu.cpp`：
- 将 RUAPU_IMPLEMENTATION 条件从 `__riscv` 改为 `0`（禁用 RISC-V 上的 ruapu）
- 移除 `ruapu_init()` 和 `ruapu_supports()` 调用
- RISC-V CPU flags 保持默认值 0

---

## 阶段二：分层量化探索（2026-05-26）

### 2.1 问题分析

ONNX 模型权重分布（193MB FP32）：

| 组件 | FP32 | 占比 |
|------|------|------|
| Conv (ResNet18) | 43.6 MB | 22.5% |
| MatMul (Transformer attention Q/K/V/O) | 101.0 MB | 52.2% |
| Gemm (Linear layers) | 12.1 MB | 6.2% |
| 其他 (LayerNorm/bias/Split) | 36.4 MB | 18.6% |

### 2.2 量化策略对比

| 方案 | 模型大小 | 平均误差(vs FP16) | 结论 |
|------|----------|------------------|------|
| 全 FP16 | 95 MB | 基准 | 256MB OOM |
| 全 INT8 (ACIQ) | 48 MB | 10.04% | 误差太大 |
| 全 INT8 (KL) | 48 MB | 10.70% | 误差太大 |
| ResNet INT8 + Transformer FP16 | **84 MB** | **0.30%** | ✅ 最优 |
| ResNet FP16 + Transformer INT8 | 59 MB | 9.79% | 无改善 |

**关键发现**：精度瓶颈在 Transformer（Gemm/MHA），不在 ResNet。逐张量激活量化是限制因素。选择性量化是有效折中。

### 2.3 最终方案：INT8-v3

- 只量化 ResNet 卷积层
- Transformer 层保持 FP16
- 校准数据：10 组 NPY 格式（图 + 状态）
- 模型：`act_det_int8_v3.{param,bin}` (84MB)

---

## 阶段三：QEMU 验证（2026-05-26）

### 3.1 512MB 验证

| 模型 | 推理时间 | 精度 | 状态 |
|------|---------|------|------|
| FP16 | ~15s | 逐位一致 | PASS |
| INT8-v3 | ~15s | 平均误差 0.30% | PASS |

### 3.2 256MB 验证

| 模型 | 大小 | 256MB QEMU | 精度(vs FP32) |
|------|------|-----------|--------------|
| FP16 | 95 MB | OOM | — |
| INT8-v3 | 84 MB | **PASS** | **平均 0.30%** |

### 3.3 精度交叉验证（50 用例，800 对比点）

| 统计量 | 绝对误差 | 相对误差 |
|--------|---------|---------|
| 平均值 | 0.0030 | 0.41%（中位数） |
| P90 | 0.0066 | 1.59% |
| P99 | 0.0150 | 9.54% |
| 最大值 | 0.0241 | — |

82.2% < 0.005，96.1% < 0.01，100% < 0.025。

---

## 阶段四：SG2002 CPU 实机验证（2026-05-26）

| 项目 | 结果 |
|------|------|
| 内核 | StarryOS SG2002, AX_LOG=warn |
| 模型 | INT8-resonly (84MB) |
| 推理时间 | ~38s (C906 标量 @ 1GHz) |
| 内存 | 峰值堆 ~63MB, 256MB 充裕 |
| 输出 | 与 x86 基线一致 |

---

## 阶段五：TPU-MLIR 编译管线（2026-05-27）

### 5.1 编译流程

```
ONNX (FP32, 194MB)
  │ model_transform.py → MLIR TOP
  ▼
MLIR TOP (GELU = Erf 多项式展开)
  │ tpuc-opt --convert-top-to-tpu
  ▼
MLIR TPU (BF16, GELU 被拆解)
  │ address 分配 + codegen
  ▼
cvimodel (96MB)
```

### 5.2 运算分布与 TPU 支持情况

| 运算 | 数量 | 权重占比 | TPU 支持方式 |
|------|------|---------|-------------|
| Conv2D | 21 | 22.5% | 原生支持 |
| MatMul (Attention Q/K/V/O) | 84 | 52.2% | 原生支持 |
| LayerNorm | 20 | — | Div → LutBF16 Slope |
| Softmax | 11 | — | Div → LutBF16 Slope |
| GELU | 18 | — | Erf 或近似替代 |
| Add/Mul/Reshape/Permute | 200+ | — | 原生支持 |

### 5.3 三种 GELU 替换策略对比（都已废弃！）

| 策略 | ONNX 变化 | Mantissa LutBF16 数 | vs FP32 原始模型误差 |
|------|----------|---------------------|-------------------|
| 原始 Erf | 9×Erf + 10×Div | 9 | 基准 |
| Clip+Rational | 9×Clip + 18×Div | 9 | max 0.414, mean 0.134 |
| Rational | 18×Div | 9 | 未测试 |
| NoErf (Clip) | 9×Clip + 9×Div | **0** | max 0.257, mean 0.058 |

**关键发现**：Clip+Rational 和 Rational 策略**并未减少 Mantissa LutBF16 数量**，只是额外增加了 Div 运算。

### 5.4 误差来源分解（NoErf 变体）

| 误差来源 | 最大误差 | 占比 | 说明 |
|---------|---------|------|------|
| GELU 近似（Clip 替代） | 0.253 | 98.4% | 单层近似误差 ≈ 0.00007，经 18 层 Transformer 传播后放大 |
| BF16 量化 | 0.005 | 1.6% | Conv/MatMul 权重和激活从 FP32 → BF16 |

**结论**：误差的 98% 来自 GELU 近似，而非 BF16 量化。

---

## 阶段六：关键突破 — 确定性 ONNX 导出（2026-05-27）

### 6.1 根因发现

**之前所有 TPU 方案误差大（max 1.92）的根因不是 BF16 量化或 GELU 近似，而是 ONNX 导出时 CVAE latent 的随机采样。**

原始导出脚本使用 `torch.randn_like(mu)` 生成 eps，导致每次导出/推理的 latent 不同，PyTorch 和 TPU 参考值不一致。

### 6.2 解决方案

1. **确定性 ONNX 导出**：monkey-patch `_sample_latent` 使 eps=0，z=mu
2. **PyTorch 遗留 C++ 导出器**：`dynamo=False`，强制 opset 14（TPU-MLIR 兼容）
3. **原始 Erf GELU**：不替换，由 TPU-MLIR 自行处理

### 6.3 结果

- TPU BF16 x86 模拟 vs PyTorch FP32 = **max 0.0048, mean 0.0030**
- 19 个 LutBF16，全部为 Slope 模式（无 Mantissa 模式）
- 误差纯属 BF16 量化，无 GELU 近似误差
- 编译管线完整走通：ONNX → model_transform.py → tpuc-opt → model_deploy.py → cvimodel

---

## 阶段七：SG2002 内核编译问题（2026-05-28）

### 7.1 问题

`dev-inference` 分支在 2026-05-27 的 rebase 后，编译出的内核（2.6MB）在 SG2002 上启动后立即乱码，无法正常 boot。

### 7.2 排查过程

**二分法定位**：

| Commit | 内核大小 | HAL 重构 | UART 改动 | Boot |
|--------|---------|----------|----------|------|
| `c550b7b85` (rebase前) | 2.4MB | 无 | 无 | ✅ |
| `edc8adec4^` | 2.48MB | 无 | 无 | — |
| `a0dbe2d2b` | 2.5MB | 有 | 无 | ❓ |
| `83cb9fca2` (HEAD) | 2.6MB | 有 | 有 | ❌ |

**根因**：HAL 重构 (`edc8adec4`) + UART 驱动替换 (`40a4fe3c1`) 等约 40 个 origin/dev 的 commit 在 rebase 时被合并入 `dev-inference`，破坏了 SG2002 的 boot。

### 7.3 解决方案

`git reset --hard c550b7b85` 回到 rebase 前的安全 commit。所有 ACT/TPU 内容（pro57、ncnn、onnxruntime、tract）均已包含在此 commit 中。备份分支：`dev-inference-backup`。

```bash
git branch dev-inference-backup dev-inference
git reset --hard c550b7b85
cargo xtask starry quick-start licheerv-nano-sg2002 build
```

编译器出的内核 2,462,016 字节，MD5=`e8309c1454073ec5121ddefca8dcbef7`，与之前能用的 `sg2002_minimal` 镜像中的内核逐字节相同。

### 7.4 SD 卡镜像拼装

SG2002 的 SD 卡镜像由以下组件组成：

```
SD 卡镜像 (~1.1GB, MBR/DOS 分区表)
├── Partition 1: FAT32 64MB (bootable)
│   ├── fip.bin    — Cvitek 一级 bootloader (441KB)
│   └── boot.sd    — FIT Image = kernel + ramdisk + DTB (~5MB)
├── Partition 2: ext4 1GB
│   └── Alpine Linux rootfs
```

**启动流程**：`ROM → fip.bin (FSBL+BL31+U-Boot SPL) → boot.sd (FIT) → StarryOS kernel`

**FIT Image (boot.sd) 内容**：

| 组件 | 说明 | 加载地址 |
|------|------|---------|
| kernel-1 | StarryOS kernel (raw binary) | 0x80200000 |
| ramdisk-1 | Cvitek initramfs (2.5MB) | — |
| fdt-xxx | 设备树 DTB (24KB) | — |

**编译命令链**：

```bash
# 1. 编译内核
cargo xtask starry quick-start licheerv-nano-sg2002 build
# → target/riscv64gc-unknown-none-elf/release/starryos.bin

# 2. 准备 FIT image (boot.sd) — 用 mkimage + ITS 文件
mkimage -f boot.its boot.sd

# 3. 创建 SD 卡镜像 (无需 sudo)
dd if=/dev/zero of=output.img bs=512 count=0 seek=2244608
sfdisk output.img <<EOF ...
mkfs.fat -F 32 -n BOOT --offset 2048 output.img
mcopy -i output.img@@$((2048*512)) fip.bin boot.sd ::
dd if=rootfs.img of=output.img bs=512 seek=133120 conv=notrunc
resize2fs output.img?offset=$((133120*512))

# 4. 注入 TPU 推理文件(debugfs,无需sudo)
debugfs -w rootfs.img <<EOF
mkdir /tpu
write act_infer_batch act_infer_batch
write inputs/* bin test_inputs/
EOF
```

---

## 阶段八：板上 TPU 批量验证（2026-05-28）

### 8.1 批量测试设计

**100 个测试用例**，覆盖多种输入分布：

| 用例范围 | 输入分布 | 说明 |
|---------|---------|------|
| 0-4 | images=zeros, state=zeros | 全零基线 |
| 5-9 | images=zeros, state=[0.5, -0.3] | 零图+固定状态 |
| 10-19 | images=uniform[-1,1], state固定 | 均匀随机图像 |
| 20-29 | images=uniform[-1,1], state=randn*0.5 | 均匀随机+小状态 |
| 30-49 | images=randn*0.5, state=randn*0.5 | 正态随机图像 |
| 50-69 | images=randn*0.1, state=randn*3.0 | 小图像+极端状态 |
| 70-84 | images=mixed, state=randn*2.0 | 混合分布 |
| 85-99 | images=structured(sin), state=uniform[-2,2] | 结构化图像 |

### 8.2 板上验证结果

```
Board TPU BF16 vs PyTorch FP32 (基线)
100 cases × 16 values = 1600 comparison points

Max:   0.119721
Mean:  0.004508
P50:   0.002603
P90:   0.010095
P95:   0.014076
P99:   0.037592

Error distribution:
  [0.000, 0.005): 75.1%
  [0.005, 0.010): 14.7%
  [0.010, 0.020):  7.7%
  [0.020, 0.030):  1.1%
  [0.030, 0.050):  1.1%
  [0.050, 1.000):  0.3%

Avg inference time: 86.14ms/case
Speedup vs CPU (38s): 441x
```

### 8.3 数据完整性审计

| 检查项 | 结果 |
|--------|------|
| 用例完整性 | 100/100，无 NaN，无 Inf |
| 相同输入一致性 | cases 0-4 完全一致，cases 5-9 完全一致 |
| 不同输入差异性 | 4005/4005 随机对全部不同 |
| 输入对齐验证 | 7 个采样点全部对齐 |

---

## 关键经验教训

1. **CVAE latent 随机性是最大陷阱**：确定性导出（eps=0）是精度验证的前提
2. **GELU 替换得不偿失**：原始 Erf GELU 的 BF16 量化误差远小于任何替换策略
3. **Git rebase 可能破坏平台支持**：origin/dev 的 HAL/UART 重构导致 SG2002 boot 失败
4. **二进制一致性验证是黄金标准**：MD5 一致的内核 = 能用的内核
5. **精度基线永远是 PyTorch FP32**：x86 TPU 模拟只是中间参考，不算基线
6. **debugfs 可在无 sudo 下操作 ext4**：注入文件到 rootfs 无需特权

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `pro57/act/modeling_act.py` | ACT PyTorch 模型定义 |
| `pro57/act/configuration_act.py` | ACT 模型配置 |
| `pro57/act/train_act.py` | ACT 训练脚本 |
| `pro57/act/act_infer.c` | 单次 TPU 推理（C，硬编码输入） |
| `pro57/act/act_infer_batch.c` | 批量 TPU 推理（C，从文件读取） |
| `final_model.pt` | PyTorch 训练权重 (194MB) |
| `act_model_det.onnx` | 确定性 ONNX 导出 (opset 14) |
| `act_model_det_cv181x_bf16.cvimodel` | TPU 编译产物 (96MB BF16) |
| `test-suit/starryos/normal/tract-inference/src/main.rs` | tract-onnx Gemm 推理验证 |
| `test-suit/starryos/normal/act-inference/src/main.rs` | tract-onnx 完整 ACT 模型推理 |
| `test_ort/build_act_test.sh` | ONNX Runtime 交叉编译脚本 |
| `test_ort/ort_test.c` | ONNX Runtime 推理程序 |
| `build_act_test.sh` | ncnn 交叉编译脚本 |
| `os/StarryOS/configs/board/licheerv-nano-sg2002.toml` | SG2002 编译配置 |
| `os/StarryOS/configs/board/licheerv-nano-sg2002.dtb` | 设备树 |
| `doc/sg2002-starryos-image-guide.md` | SD 卡镜像拼装手册 |
| `doc/act-tpu-reproduce.md` | 复现手册 |
| `doc/act-tpu-records.md` | 测试记录 |
| `tpu-board-100cases-rawdata` | 100 用例板上原始输出数据 |
