# ACT 模型推理移植进度

日期：2026-05-09

## 已完成项

### 1. tract-onnx 推理验证（5/5）
- tract-hello：12 个测试用例交叉验证（zeros/ones/10 random）
- simple.onnx（571KB）在 StarryOS RISC-V QEMU 512MB PASS
- 经验：tract 内存占用高，不支持 INT8 量化

### 2. onnxruntime 1.21.0 交叉编译（5/7）
- 工具链：`riscv64-linux-gnu-g++` 13.3.0
- 修改 3 处 cmake 配置
- 最终产物：12MB 静态链接 RISC-V 二进制
- ONNX → ORT 模型转换：193MB → 188MB

### 3. Conv/BN 诊断测试（5/8 上午）
- 构造最小 Conv 测试模型验证 RISC-V 浮点计算正确性
- conv1x1：27/27 完美匹配（0 ULP 差异）
- conv3x3：6/9 完美匹配，3/9 差 1 ULP
- conv+BatchNorm：18/18 完美匹配
- **结论：RISC-V 推理结果与 x86 一致**

### 4. mmap 内存优化 + 512MB 验证（5/8 中午）
- mmap 加载 ORT 模型（MAP_PRIVATE + fd）
- 实测堆内存 ~258MB，512MB QEMU 推理通过
- 256MB 仍然 OOM

### 5. ncnn 框架迁移（5/8 下午）
- 从 ORT 切换到 ncnn（轻量级、原生 FP16/INT8 支持）
- pnnx 模型转换：PyTorch → ncnn（95MB FP16 / 48MB INT8）
- FP16 模型 512MB 推理通过（堆峰值 ~205MB）

### 6. ncnn INT8 量化精度优化（5/9）✅
- 探索 4 个 INT8 优化方案
- **最终方案：ResNet INT8 + Transformer FP16**
  - 模型大小：84MB（比全 FP16 节省 11MB）
  - 平均误差：0.19%（vs FP16 参考）
  - 堆峰值：63MB（256MB 内充裕）
  - RISC-V QEMU 256MB 验证 PASSED

## 已解决问题

### 1. printf float 格式化 bug
- RISC-V 静态链接 glibc 下 `printf("%f", ...)` 输出垃圾值
- 解决：用 hex 打印 + 主机 python 解码验证

### 2. zero_copy_for_initializers COW panic
- ORT 将权重标记为只读，ACT 模型某些 Op 会写入输入张量
- 解决：关闭 zero_copy_for_initializers

### 3. xtask starry test 注入文件源路径
- 更新注入文件必须改 `sh/` 源目录，不是 `target/.../overlay/`

### 4. INT8 全量化精度损失 10%
- 根本原因：逐张量激活量化（每个层一个 scale），无法兼顾不同通道的动态范围
- 解决：选择性量化（ResNet INT8，Transformer FP16）

## 待解决问题

### 1. Transformer INT8 量化精度修复
- Gemm/MultiHeadAttention INT8 量化是精度瓶颈
- 嫌疑点：`out_weight_data_int8_scale` 和 `B_data_int8_scale` 的逐张量量化
- 如需 48MB 以下模型，必须解决此问题

### 2. 荔枝派 SG2002 实际部署
- QEMU 下 256MB 通过，但实际板子可能环境不同
- 需在硬件上实测推理延迟和内存占用

### 3. 推理性能优化
- 当前 RISC-V QEMU 推理耗时 ~48s
- 可能的优化方向：多线程、算子融合、内存池

## 文件说明

### 测试程序
- `tract-inference/src/main.rs` — tract-onnx 推理验证（12 测试用例）
- `ort_test.c` — ONNX Runtime C API 测试（含 brk 内存追踪）
- `ort_conv_test.c` — Conv1x1/3x3/BN 诊断测试（跨平台浮点对比）
- `float_diag.c` — 基本浮点运算诊断
- `mmap_test.c` — MAP_PRIVATE mmap 正确性验证
- `ncnn_test.c` — ncnn 推理测试（含 brk 内存追踪）

### 模型转换工具
- `act_det_pnnx.py` — PyTorch 模型定义 + pnnx 导出
- `ncnn2table` — 校准数据 → 量化表生成
- `ncnn2int8` — FP16 模型 → INT8 量化（含选择性量化逻辑）

### QEMU 测试用例
- `test-suit/starryos/normal/tract-inference/` — tract-onnx 测试
- `test-suit/starryos/normal/ort-inference/` — ORT ACT 模型测试
- `test-suit/starryos/normal/ort-conv-test/` — Conv/BN 诊断测试
- `test-suit/starryos/normal/float-diag/` — 浮点诊断测试
- `test-suit/starryos/normal/mmap-test/` — mmap 验证测试
- `test-suit/starryos/normal/ncnn-inference/` — ncnn INT8 选择性量化测试（84MB 模型，256MB）

### 文档
- `doc/5.9-1.md` — 全流程记录（tract → ORT → ncnn INT8 优化）
- `doc/ort-progress.md` — 本文件（进度追踪）
