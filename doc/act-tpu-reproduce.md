# ACT 模型 TPU 推理 — 复现手册

本文档提供从零开始复现 SG2002 板上 ACT 模型 TPU 推理的完整步骤。

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
| `riscv64-linux-gnu-gcc/g++` | 交叉编译 TPU 推理程序 | 系统安装 |
| `mkimage` (u-boot-tools) | 生成 FIT image | `apt install u-boot-tools` |
| `mtools` (mformat/mcopy) | 操作 FAT32 分区 | `apt install mtools` |
| `mkfs.ext4` | 格式化 ext4 分区 | 系统自带 |
| `sfdisk` | 创建 MBR 分区表 | 系统自带 |
| `debugfs` | 操作 ext4 文件系统 | 系统自带 |
| `python3` + PyTorch | 运行 PT FP32 基线推理 | `pip install torch numpy` |

### 项目文件

| 文件 | 说明 | 来源 |
|------|------|------|
| `final_model.pt` | PyTorch 训练权重 | 训练产出 |
| `act_model_det_cv181x_bf16.cvimodel` | TPU 编译模型 | TPU-MLIR 编译 |
| `tpu_sdk/` | Cvitek TPU SDK | 外部 SDK |
| `LicheeRV-Nano-Build/` | Cvitek 厂商 SDK | 外部 SDK |

---

## 步骤一：编译 StarryOS 内核

```bash
# 确认在正确的 commit（rebase 之前的安全版本）
git checkout dev-inference
git log --oneline -1
# 应该显示: c550b7b85 docs(devlog): add ACT model inference development log

# 编译内核
cargo xtask starry quick-start licheerv-nano-sg2002 build

# 产出文件
ls -lh target/riscv64gc-unknown-none-elf/release/starryos.bin
# 应该是 2,462,016 字节
# MD5: e8309c1454073ec5121ddefca8dcbef7
```

---

## 步骤二：准备 SD 卡组件

### 2.1 获取 fip.bin 和 ramdisk

从已知可用的 SG2002 镜像中提取（这些文件不随内核变化）：

```bash
# 假设 work.img 是已知可用的镜像
dd if=work.img bs=512 skip=2048 count=131072 of=boot_part.img
mcopy -i boot_part.img ::/fip.bin .
# 提取 ramdisk
python3 extract_ramdisk.py boot.sd  # 见附录
```

### 2.2 准备 Rootfs

```bash
cargo xtask starry rootfs --arch riscv64
# rootfs 位于: tmp/axbuild/rootfs/rootfs-riscv64-alpine.img
```

### 2.3 编写 ITS 文件

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

### 2.4 生成 FIT Image

```bash
cp target/riscv64gc-unknown-none-elf/release/starryos.bin .
cp os/StarryOS/configs/board/licheerv-nano-sg2002.dtb .
mkimage -f boot.its boot.sd
```

---

## 步骤三：组装 SD 卡镜像

```bash
#!/bin/bash
# 创建 1.1GB 空镜像
OUT="sg2002_tpu_$(date +%Y%m%d_%H%M%S).img"
dd if=/dev/zero of="$OUT" bs=512 count=0 seek=2244608

# 创建 MBR 分区表 (P1: FAT32 64MB, P2: ext4 1GB)
sfdisk "$OUT" <<EOF
label: dos
unit: sectors
start=2048, size=131072, type=c, bootable
start=133120, size=2111488, type=83
EOF

# 格式化 boot 分区并复制文件
mkfs.fat -F 32 -n BOOT --offset 2048 "$OUT"
mcopy -i "$OUT"@@$((2048*512)) fip.bin boot.sd ::

# 复制 rootfs 到分区 2
dd if=rootfs-riscv64-alpine.img of="$OUT" bs=512 seek=133120 conv=notrunc
resize2fs "${OUT}?offset=$((133120*512))"
```

---

## 步骤四：交叉编译 TPU 推理程序

```bash
riscv64-linux-gnu-g++ -static -O2 -march=rv64gc -mabi=lp64d \
  -I tpu_sdk/cviruntime/include \
  -o act_infer_batch \
  pro57/act/act_infer_batch.c \
  tpu_sdk/build/cviruntime/src/soc/181x/libcviruntime-static.a \
  tpu_sdk/build/cvikernel/libcvikernel-static.a \
  -lpthread -lm -lstdc++
```

产出 `act_infer_batch` (ELF, 静态链接 RISC-V)。

---

## 步骤五：注入 TPU 推理文件

```bash
# 复制 rootfs 用于编辑
cp rootfs-riscv64-alpine.img rootfs_tpu.img

# 注入文件 (无需 sudo)
debugfs -w rootfs_tpu.img <<EOF
mkdir /tpu
cd /tpu
write act_infer_batch act_infer_batch
write act_model_det_cv181x_bf16.cvimodel act_model.cvimodel
mkdir test_inputs
$(for f in inputs/*.bin inputs/count.txt; do echo "write $f $(basename $f)"; done)
EOF

# 用编辑过的 rootfs 重新组装镜像（重复步骤三的 dd 步骤）
dd if=rootfs_tpu.img of="$OUT" bs=512 seek=133120 conv=notrunc
resize2fs "${OUT}?offset=$((133120*512))"
```

---

## 步骤六：烧录与运行

```bash
# 烧录 SD 卡
sudo dd if=sg2002_tpu_*.img of=/dev/mmcblk0 bs=4M status=progress conv=fsync

# 插入 SG2002，上电，串口登录（115200 8N1）

# 板上运行
/tpu/act_infer_batch /tpu/act_model.cvimodel /tpu/test_inputs /tpu/test_outputs
```

输出格式（单行紧凑格式）：
```
C0 86.1 -1.156250 0.051270 -0.707031 -0.449219 -0.531250 -0.621094 -0.453125 -0.714844 -0.472656 -0.707031 -0.458984 -0.707031 -0.462891 -0.679688 -0.447266 -0.652344
C1 86.1 0.578125 -1.664062 0.018433 -1.101562 -0.133789 -0.949219 -0.455078 -0.632812 -0.531250 -0.578125 -0.535156 -0.570312 -0.382812 -0.722656 -0.281250 -0.777344
...
Done. Total: 8438.96 ms, Avg: 84.39 ms
```

---

## 步骤七：精度验证

```bash
# 在开发机上运行 PyTorch FP32 基线推理
# 使用相同的随机种子 (seed=42) 和相同的输入分布
python3 run_pt_baseline.py  # → pt_ref.npz

# 对比板上输出
python3 compare.py board_output.txt pt_ref.npz
```

期望结果：
```
Board TPU BF16 vs PyTorch FP32 (BASELINE)
100 cases × 16 values = 1600 comparison points
Max:   ~0.12
Mean:  ~0.005
P50:   ~0.003
P90:   ~0.01
Speedup vs CPU: 441x
```

---

## 常见问题

### 内核启动乱码
确认 `dev-inference` 分支在 `c550b7b85` commit，不是 rebase 后的 HEAD。

### U-Boot 找不到 boot.sd
确认 FAT32 分区中文件名是 `boot.sd`（8.3 格式），不是 `boot_v2.sd`。

### TPU 推理 SIGILL
cviruntime 的 C++ 清理析构在 RISC-V 上可能触发未实现指令。不影响推理结果。

### 串口输出截断
代码已使用单行紧凑格式（`C<n> <ms> <v0> ... <v15>`），每个 case 一行。

---

## 附录：ramdisk 提取脚本

```python
import struct

with open('boot.sd', 'rb') as f:
    data = f.read()

magic, totalsize, off_dt_struct, off_dt_strings = struct.unpack('>IIII', data[:16])

def read_u32(d, off):
    return struct.unpack('>I', d[off:off+4])[0]

pos = off_dt_struct
struct_end = off_dt_strings
path = []

while pos < struct_end:
    token = read_u32(data, pos); pos += 4
    if token == 0x00000001:  # FDT_BEGIN_NODE
        end = data.index(0, pos)
        path.append(data[pos:end].decode())
        pos = (end + 4) & ~3
    elif token == 0x00000002:  # FDT_END_NODE
        path.pop()
    elif token == 0x00000003:  # FDT_PROP
        prop_len = read_u32(data, pos)
        nameoff = read_u32(data, pos + 4)
        pos += 8
        # read property name
        end = data.index(0, off_dt_strings + nameoff)
        name = data[off_dt_strings + nameoff:end].decode()
        full = '/'.join(path)
        if 'ramdisk' in full and name == 'data':
            ramdisk = data[pos:pos + prop_len]
            with open('cvitek-ramdisk.gz', 'wb') as f:
                f.write(ramdisk)
            print(f'Extracted: {prop_len} bytes')
        pos = (pos + prop_len + 3) & ~3
    elif token == 0x00000009:  # FDT_END
        break
```
