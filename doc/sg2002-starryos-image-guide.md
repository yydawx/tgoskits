# SG2002 StarryOS SD 卡镜像编译与拼装手册

## 概述

SG2002（荔枝派 LicheeRV Nano）的 StarryOS SD 卡镜像由以下组件组成：

```
SD 卡镜像 (~1.1GB, MBR/DOS 分区表)
├── Sector 0-2047:  MBR + 预留空间 (1MB)
├── Partition 1:    FAT32 64MB (bootable, type 0x0C)
│   ├── fip.bin       Cvitek 一级 bootloader (441KB)
│   └── boot.sd       FIT Image = kernel + ramdisk + DTB (~5MB)
├── Partition 2:    ext4 1GB (type 0x83)
│   └── Alpine Linux rootfs
```

**启动流程**：`ROM → fip.bin (FSBL+BL31+U-Boot SPL) → boot.sd (FIT) → StarryOS kernel`

---

## 准备工作

### 环境要求

本机上需要以下工具：

| 工具 | 用途 | 安装 |
|------|------|------|
| `cargo xtask` | 编译 StarryOS kernel | 项目自带 |
| `mkimage` | 生成 FIT image | `apt install u-boot-tools` |
| `mformat, mcopy` | 创建/操作 FAT32 分区 | `apt install mtools` |
| `mkfs.ext4` | 格式化 ext4 分区 | 系统自带 |
| `fdisk` | 创建 MBR 分区表 | 系统自带 |
| `dd` | 读写原始磁盘数据 | 系统自带 |

### 关键路径速查

| 路径 | 说明 |
|------|------|
| `os/StarryOS/configs/board/licheerv-nano-sg2002.toml` | SG2002 编译配置 |
| `os/StarryOS/configs/board/licheerv-nano-sg2002.dtb` | 设备树 (DTB) |
| `os/StarryOS/configs/board/licheerv-nano-sg2002-uboot.toml` | U-Boot 串口配置 |
| `target/riscv64gc-unknown-none-elf/release/starryos.bin` | 编译产物（raw kernel） |
| `target/riscv64gc-unknown-none-elf/release/starryos.uimg` | 编译产物（U-Boot 格式） |
| `tmp/axbuild/rootfs/rootfs-riscv64-alpine.img` | 托管 rootfs 镜像 |
| `target/sg2002/` | 最终 SD 卡镜像存放目录 |

---

## 步骤一：编译 StarryOS 内核

```bash
cargo xtask starry quick-start licheerv-nano-sg2002 build
```

产出文件：
- `target/riscv64gc-unknown-none-elf/release/starryos.bin` — 原始内核二进制（2.6MB）
- `target/riscv64gc-unknown-none-elf/release/starryos.uimg` — U-Boot 格式内核（带 64 字节头）
- 加载地址：`0x80200000`

**注意**：FIT image 打包时需要 `.bin`（raw binary），不是 `.uimg`。

---

## 步骤二：准备 Rootfs

StarryOS 使用受管 rootfs，首次使用会自动下载：

```bash
cargo xtask starry rootfs --arch riscv64
```

下载后 rootfs 位于：`tmp/axbuild/rootfs/rootfs-riscv64-alpine.img`

---

## 步骤三：准备 fip.bin

`fip.bin` 是 Cvitek 的一级 bootloader（FSBL + ATF BL31 + U-Boot SPL），不随内核变化，可复用。

### 方案 A：从已有镜像提取（推荐）

```bash
# 从已知可用的镜像中提取
mcopy -i <boot分区> ::/fip.bin fip.bin

# 验证
md5sum fip.bin
# 已知可用的大小: 440,832 bytes
```

### 方案 B：从 vendor SDK 编译（需要 riscv64 交叉工具链）

```bash
cd LicheeRV-Nano-Build
source build/envsetup_soc.sh
defconfig sg2002_licheervnano_sd
make fsbl
# 产出: fsbl/build/<platform>/fip.bin
```

---

## 步骤四：生成 FIT Image (boot.sd)

FIT Image 是一个打包了 kernel + ramdisk + DTB 的扁平设备树镜像。

### 4.1 理解 FIT Image 结构

以已知可用镜像的 boot.sd 为例：

```
FIT Image (boot.sd, ~5MB)
├── kernel-1       StarryOS kernel, 2.4MB, load=0x80200000
├── ramdisk-1      cvitek initramfs, 2.5MB
└── fdt-xxx        设备树 DTB, 24KB
```

### 4.2 编写 ITS 文件

创建 `boot.its`（Image Tree Source）：

```dts
/dts-v1/;

/ {
    description = "StarryOS kernel for SG2002 LicheeRV Nano";
    #address-cells = <1>;

    images {
        kernel-1 {
            description = "StarryOS kernel";
            data = /incbin/("starryos.bin");
            type = "kernel";
            arch = "riscv";
            os = "linux";
            compression = "none";
            load = <0x80200000>;
            entry = <0x80200000>;
            hash-1 {
                algo = "crc32";
            };
        };

        ramdisk-1 {
            description = "cvitek ramdisk";
            data = /incbin/("cvitek-ramdisk.gz");
            type = "ramdisk";
            arch = "riscv";
            os = "linux";
            compression = "none";
            hash-1 {
                algo = "crc32";
            };
        };

        fdt-sg2002_licheervnano_sd {
            description = "cvitek device tree - sg2002_licheervnano_sd";
            data = /incbin/("licheerv-nano-sg2002.dtb");
            type = "flat_dt";
            arch = "riscv";
            compression = "none";
            hash-1 {
                algo = "sha256";
            };
        };
    };

    configurations {
        default = "config-sg2002_licheervnano_sd";
        config-sg2002_licheervnano_sd {
            description = "StarryOS boot for sg2002_licheervnano_sd";
            kernel = "kernel-1";
            ramdisk = "ramdisk-1";
            fdt = "fdt-sg2002_licheervnano_sd";
        };
    };
};
```

### 4.3 生成 boot.sd

```bash
# 将 starryos.bin 和 DTB 复制到工作目录
cp target/riscv64gc-unknown-none-elf/release/starryos.bin .
cp os/StarryOS/configs/board/licheerv-nano-sg2002.dtb .

# 提取 ramdisk（从已有镜像中）
mkimage -l boot.sd          # 查看已有 FIT 中的组件
# 通过 fdtdump 或用 Python 脚本提取 ramdisk-1 数据

# 生成 FIT image
mkimage -f boot.its boot.sd
```

### 4.4 提取已有 ramdisk（如果复用）

```bash
# 方法：用 Python 从 FIT image 中提取 ramdisk
python3 -c "
import struct
with open('boot.sd.from.working', 'rb') as f:
    data = f.read()
# FIT image 是 DTB 格式，可用 fdtdump 获取偏移
# ramdisk-1 的 data 在 images/ramdisk-1/data 属性中
"

# 或者直接用 dd 跳过 kernel-1 取出 ramdisk-1
# 具体偏移通过 fdtdump 查看
```

---

## 步骤五：组装 SD 卡镜像

使用 `mtools`（mformat/mcopy）和 `fdisk` 可在**不需要 sudo** 的情况下创建镜像。

### 5.1 创建空镜像 + 分区表

```bash
# 创建 1.1GB 空镜像（2,244,608 扇区）
dd if=/dev/zero of=sg2002_starryos.img bs=512 count=0 seek=2244608

# 创建 MBR 分区表（非交互式）
# 分区 1: 2048-133119, FAT32, bootable
# 分区 2: 133120-end, ext4
fdisk sg2002_starryos.img <<EOF
n
p
1
2048
133119
n
p
2
133120

a
1
t
1
c
t
2
83
w
EOF
```

### 5.2 创建 FAT32 boot 分区

```bash
# mtools 配置文件
cat > sg2002.mtoolsrc <<EOF
drive b: file="sg2002_starryos.img" partition=1
EOF

export MTOOLSRC=sg2002.mtoolsrc

# 格式化 FAT32
mformat -v BOOT -h 255 -s 63 -t 8192 b:

# 复制文件
mcopy fip.bin b::
mcopy boot.sd b::
```

### 5.3 创建 ext4 rootfs 分区

```bash
# mtools 配置文件
cat > mtoolsrc.ext4 <<EOF
drive c: file="sg2002_starryos.img" partition=2
EOF

# 格式化 ext4
mkfs.ext4 -L rootfs -E offset=$((133120 * 512)) sg2002_starryos.img 2111488

# 挂载并复制 rootfs（需要 sudo）
sudo mount -o loop,offset=$((133120 * 512)) sg2002_starryos.img /mnt/sg2002
sudo tar xf rootfs.tar -C /mnt/sg2002
sudo umount /mnt/sg2002
```

### 5.4 验证镜像

```bash
# 检查分区表
fdisk -l sg2002_starryos.img

# 检查 boot 分区内容
mcopy -i sg2002_starryos.img@@1S ::

# 检查 FIT image
mkimage -l boot.sd
```

---

## 快速重建流程（仅更新内核）

如果只是修改了 StarryOS 内核代码，不需要重新拼装整个镜像：

```bash
# 1. 编译新内核
cargo xtask starry quick-start licheerv-nano-sg2002 build

# 2. 提取旧 ramdisk（一次性操作）
# 如果还没提取，从已知可用镜像的 boot.sd 中提取 ramdisk-1

# 3. 重新生成 boot.sd
cp target/riscv64gc-unknown-none-elf/release/starryos.bin .
mkimage -f boot.its boot.sd

# 4. 替换 boot 分区中的 boot.sd
mcopy -i sg2002_starryos.img@@1S boot.sd ::

# 5. 写入 SD 卡
dd if=sg2002_starryos.img of=/dev/mmcblk0 bs=4M status=progress
```

---

## 烧写镜像到 SD 卡

```bash
# 确认 SD 卡设备（假设为 /dev/mmcblk0）
lsblk

# 写入
sudo dd if=sg2002_starryos.img of=/dev/mmcblk0 bs=4M status=progress conv=fsync

# 验证
sync
sudo partprobe /dev/mmcblk0
```

---

## 串口启动验证

```bash
# 串口登录
screen /dev/ttyUSB0 115200

# 或通过 xtask 自动运行（需要 ostool-server 在板子上）
cargo xtask starry quick-start licheerv-nano-sg2002 run --serial /dev/ttyUSB0 --baud 115200
```

---

## 常见问题

### boot.sd 大小异常
FIT image 大小 = 内核大小 + ramdisk 大小 + DTB 大小 + FIT 开销。通常 4-6MB。

### 分区偏移计算
- 分区 1 起始偏移：`2048 × 512 = 1,048,576` (1MB)
- 分区 2 起始偏移：`133120 × 512 = 68,157,440` (65MB)
- 分区 1 大小：`131072 × 512 = 67,108,864` (64MB)

### fdisk 非交互模式
`fdisk` 支持 heredoc 输入，无需 `expect`。如果镜像很大导致 fdisk 计算默认值不准，手动指定扇区值。

### ramdisk 是否需要修改
通常不需要。ramdisk 是 Cvitek 平台初始化用的 initramfs，与 StarryOS 内核版本无关，可复用。
