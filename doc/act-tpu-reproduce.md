# ACT 模型 TPU 推理 — 复现手册

从 PyTorch 模型到 SG2002 板上 TPU 推理的完整流程，包含 BF16/INT8/混合精度量化及多种剪枝方案。

## 环境要求

### 硬件
- LicheeRV Nano (SG2002, 256MB DRAM)
- SD 卡 (≥2GB)
- USB-TTL 串口线 (115200 8N1)
- x86 Linux 开发机

### 软件
| 工具 | 用途 | 安装 |
|------|------|------|
| `cargo xtask` | 编译 StarryOS kernel | 项目自带 |
| `riscv64-linux-gnu-g++` | 交叉编译 TPU 推理程序 | 系统安装 |
| `mkimage` | 生成 FIT image | `apt install u-boot-tools` |
| `mtools` | 操作 FAT32 分区 | `apt install mtools` |
| `debugfs` | 操作 ext4 文件系统 | 系统自带 |
| `sfdisk` | 创建 MBR 分区表 | 系统自带 |
| TPU-MLIR | ONNX → cvimodel 编译 | `pip install tpu_mlir` |
| `python3` + PyTorch | 模型剪枝/校准/基线推理 | `pip install torch numpy onnx` |

### 项目文件
| 文件 | 说明 |
|------|------|
| `final_model.pt` | PyTorch 训练权重 (194MB) |
| `os/StarryOS/configs/board/licheerv-nano-sg2002.toml` | SG2002 编译配置 |
| `os/StarryOS/configs/board/licheerv-nano-sg2002.dtb` | 设备树 |
| `pro57/prune/prune_all.py` | 模型剪枝工具 |
| `pro57/prune/bench_multi.c` | 多模型 TPU 推理程序 |
| `tpu_sdk/` | Cvitek TPU SDK (cviruntime) |
| `LicheeRV-Nano-Build/` | Cvitek 厂商 SDK (fip.bin, ramdisk) |

---

## 步骤一：编译 StarryOS 内核

```bash
git checkout dev-tpu-inference
cargo xtask starry quick-start licheerv-nano-sg2002 build

# 产出: target/riscv64gc-unknown-none-elf/release/starryos.bin
# 大小: 2,462,016 bytes
# ION/TPU 已编译在内
```

---

## 步骤二：TPU-MLIR 模型编译

### 2.1 确定性 ONNX 导出

CVAE latent 必须确定性导出（eps=0），否则每次推理结果不同。

```python
import torch
from pro57.act.modeling_act import ACTModel
from pro57.act.configuration_act import ACTConfig

ckpt = torch.load('final_model.pt', map_location='cpu', weights_only=False)
cfg = ACTConfig(**ckpt['config'])
model = ACTModel(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# 确定性 CVAE: eps=0, z=mu
model.set_inference_latent(ckpt['inference_latent_mu'], ckpt['inference_latent_log_sigma'])
model._sample_latent = lambda mu, ls: mu

torch.onnx.export(model, (torch.zeros(1,3,224,224), torch.zeros(1,2)),
    'act_model_det.onnx', input_names=['images','state'], output_names=['action'],
    opset_version=14, dynamo=False)
```

### 2.2 BF16 基线模型

```bash
# ONNX → Top MLIR
model_transform.py --model_name act_bf16 --model_def act_model_det.onnx \
    --input_shapes [[1,3,224,224],[1,2]] --mlir act_bf16.mlir

# Top MLIR → cvimodel (BF16)
model_deploy.py --mlir act_bf16.mlir --quantize BF16 --chip cv181x \
    --model act_bf16.cvimodel

# 产出: act_bf16.cvimodel (~96MB)
```

### 2.3 INT8 全量化模型

需要先生成校准数据（至少 10-20 组输入）：

```python
import numpy as np
rng = np.random.RandomState(42)
for i in range(20):
    images = rng.randn(1, 3, 224, 224).astype(np.float32) * 0.5
    state = rng.randn(1, 2).astype(np.float32) * 0.5
    np.savez(f'calib_{i:03d}.npz', images=images, state=state)
```

生成校准表并编译：

```bash
# 生成校准表
run_calibration.py act_bf16.mlir \
    --dataset calib_*.npz \
    --calibration_table act_cali_table

# 编译 INT8 模型
model_deploy.py --mlir act_bf16.mlir --quantize INT8 \
    --calibration_table act_cali_table --chip cv181x \
    --model act_int8.cvimodel

# 产出: act_int8.cvimodel (~48MB)
# 注意: 全 INT8 精度较差 (mean error ~0.32)，不推荐
```

### 2.4 混合精度模型（ResNet INT8 + Transformer BF16）

创建混合精度量化表——只有 backbone 层用 INT8：

```python
# 从校准表提取所有 tensor 名
# backbone 层 → INT8, 其余 → BF16
# 生成 mix_quant_table
```

```bash
model_deploy.py --mlir act_bf16.mlir --quantize INT8 \
    --calibration_table act_cali_table \
    --quantize_table mix_quant_table --chip cv181x \
    --model act_mix.cvimodel

# 产出: act_mix.cvimodel (~85MB)
# 精度: mean error ~0.019 (接近 BF16)
```

### 2.5 精度对比（x86 TPU 模拟器）

```python
import pyruntime_cvi, numpy as np
model = pyruntime_cvi.Model('act_bf16.cvimodel')
# ... 跑 100 个测试用例，对比 PyTorch FP32
```

| 方案 | 大小 | 延迟 | mean error |
|------|------|------|-----------|
| BF16 | 96MB | 86ms | 0.005 |
| Mix INT8+BF16 | 85MB | 124ms | 0.019 |
| Full INT8 | 48MB | ? | 0.324 |

---

## 步骤三：模型剪枝

```bash
# 生成所有单策略剪枝变体 + 导出 ONNX
python3 pro57/prune/prune_all.py --ckpt final_model.pt --outdir pruned_models

# 编译每个变体为 cvimodel
source sophgo-tpumilr/envsetup.sh
for onnx in pruned_models/act_enc*.onnx pruned_models/act_ffn*.onnx; do
    name=$(basename $onnx .onnx)
    model_transform.py --model_name $name --model_def $onnx \
        --input_shapes [[1,3,224,224],[1,2]] --mlir ${name}.mlir
    model_deploy.py --mlir ${name}.mlir --quantize BF16 --chip cv181x \
        --model ${name}.cvimodel
done
```

剪枝策略和 PT 精度影响：

| 变体 | 策略 | PT diff | 说明 |
|------|------|---------|------|
| enc3 | Encoder 3层 | 0.08 | 推荐，快 7.4% |
| enc2 | Encoder 2层 | 0.14 | 更激进 |
| enc1 | Encoder 1层 | 0.25 | 精度降幅大 |
| ffn1600 | FFN 1600 | 0.38 | — |
| ffn800 | FFN 800 | 0.92 | 精度不可用 |

**核心结论：Decoder 不能剪，Encoder 可以。**

---

## 步骤四：交叉编译 TPU 推理程序

```bash
riscv64-linux-gnu-g++ -static -O2 -march=rv64gc -mabi=lp64d \
    -I tpu_sdk/cviruntime/include \
    -o bench_multi \
    pro57/prune/bench_multi.c \
    tpu_sdk/build/cviruntime/src/soc/181x/libcviruntime-static.a \
    tpu_sdk/build/cvikernel/libcvikernel-static.a \
    -lpthread -lm -lstdc++
```

---

## 步骤五：准备 SD 卡组件

### 5.1 获取 fip.bin 和 ramdisk

从已知可用的 SG2002 镜像提取（不随内核变化）：

```bash
dd if=work.img bs=512 skip=2048 count=131072 of=boot_part.img
mcopy -i boot_part.img ::/fip.bin .
# ramdisk 提取见附录
```

### 5.2 准备 Rootfs

```bash
cargo xtask starry rootfs --arch riscv64
# → tmp/axbuild/rootfs/rootfs-riscv64-alpine.img
```

### 5.3 生成 FIT Image (boot.sd)

创建 `boot.its`：

```dts
/dts-v1/;
/ {
    description = "StarryOS kernel for SG2002 LicheeRV Nano";
    #address-cells = <1>;
    images {
        kernel-1 {
            description = "StarryOS kernel";
            data = /incbin/("starryos.bin");
            type = "kernel"; arch = "riscv"; os = "linux";
            compression = "none";
            load = <0x80200000>; entry = <0x80200000>;
            hash-1 { algo = "crc32"; };
        };
        ramdisk-1 {
            description = "cvitek ramdisk";
            data = /incbin/("cvitek-ramdisk.gz");
            type = "ramdisk"; arch = "riscv"; os = "linux";
            compression = "none";
            hash-1 { algo = "crc32"; };
        };
        fdt-sg2002_licheervnano_sd {
            description = "cvitek device tree";
            data = /incbin/("licheerv-nano-sg2002.dtb");
            type = "flat_dt"; arch = "riscv";
            compression = "none";
            hash-1 { algo = "sha256"; };
        };
    };
    configurations {
        default = "config-sg2002_licheervnano_sd";
        config-sg2002_licheervnano_sd {
            description = "StarryOS boot for sg2002_licheervnano_sd";
            kernel = "kernel-1"; ramdisk = "ramdisk-1";
            fdt = "fdt-sg2002_licheervnano_sd";
        };
    };
};
```

```bash
cp target/riscv64gc-unknown-none-elf/release/starryos.bin .
cp os/StarryOS/configs/board/licheerv-nano-sg2002.dtb .
mkimage -f boot.its boot.sd
```

---

## 步骤六：组装 SD 卡镜像

全程无需 sudo：

```bash
OUT="sg2002_tpu.img"
# 创建 1.1GB 空镜像
dd if=/dev/zero of="$OUT" bs=512 count=0 seek=2244608

# MBR 分区表 (P1: FAT32 64MB, P2: ext4 1GB)
sfdisk "$OUT" <<EOF
label: dos
unit: sectors
start=2048, size=131072, type=c, bootable
start=133120, size=2111488, type=83
EOF

# 格式化 boot 分区，复制文件
mkfs.fat -F 32 -n BOOT --offset 2048 "$OUT"
mcopy -i "$OUT"@@$((2048*512)) fip.bin boot.sd ::

# 注入推理程序和模型到 rootfs
cp rootfs-riscv64-alpine.img rootfs_tpu.img
debugfs -w rootfs_tpu.img <<END
mkdir /tpu
cd /tpu
write bench_multi bench_multi
write act_bf16.cvimodel act_model.cvimodel
mkdir models
cd models
$(for m in act_enc3.cvimodel act_enc2.cvimodel act_enc1.cvimodel \
            act_ffn1600.cvimodel act_ffn800.cvimodel; do
    echo "write $m $m"
done)
mkdir test_inputs
cd test_inputs
$(for f in inputs/*.bin inputs/count.txt; do
    echo "write $f $(basename $f)"
done)
END

# 复制 rootfs 到分区 2
dd if=rootfs_tpu.img of="$OUT" bs=512 seek=133120 conv=notrunc
resize2fs "${OUT}?offset=$((133120*512))"
```

---

## 步骤七：烧录与测试

```bash
sudo dd if=sg2002_tpu.img of=/dev/mmcblk0 bs=4M status=progress conv=fsync
```

串口登录 (115200 8N1)，板上运行：

```bash
# 单模型基准测试
/tpu/bench_multi /tpu/test_inputs /tpu/act_model.cvimodel

# 多模型对比
/tpu/bench_multi /tpu/test_inputs \
    /tpu/act_model.cvimodel \
    /tpu/models/act_enc3.cvimodel \
    /tpu/models/act_enc2.cvimodel \
    /tpu/models/act_enc1.cvimodel \
    /tpu/models/act_ffn1600.cvimodel \
    /tpu/models/act_ffn800.cvimodel
```

输出格式（每行一个用例，16 个值）：
```
M0 C0 86.1 -1.1562 0.0513 -0.7070 -0.4492 ... -0.6523
M0 C1 86.1 0.5781 -1.6641 0.0184 -1.1016 ... -0.7773
```

精度基线对比（开发机上）：

```python
import numpy as np
# 跑 PyTorch FP32 基线（相同输入）
# 对比板上 TPU 输出
# 期望: BF16 max err ~0.12, mean ~0.005
```

---

## 附录：ramdisk 提取脚本

```python
import struct

with open('boot.sd', 'rb') as f:
    data = f.read()

magic, totalsize, off_dt_struct, off_dt_strings = struct.unpack('>IIII', data[:16])

def read_u32(d, off):
    return struct.unpack('>I', d[off:off+4])[0]

pos, path = off_dt_struct, []
while pos < off_dt_strings:
    token = read_u32(data, pos); pos += 4
    if token == 1:  # BEGIN_NODE
        end = data.index(0, pos)
        path.append(data[pos:end].decode())
        pos = (end + 4) & ~3
    elif token == 2:  # END_NODE
        path.pop()
    elif token == 3:  # PROP
        plen = read_u32(data, pos)
        noff = read_u32(data, pos+4); pos += 8
        end = data.index(0, off_dt_strings + noff)
        name = data[off_dt_strings + noff:end].decode()
        if 'ramdisk' in '/'.join(path) and name == 'data':
            with open('cvitek-ramdisk.gz', 'wb') as f:
                f.write(data[pos:pos+plen])
        pos = (pos + plen + 3) & ~3
    elif token == 9: break
```
