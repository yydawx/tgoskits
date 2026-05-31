# TPU-MLIR 开发参考手册

来源: https://doc.sophgo.com/sdk-docs/v26.03.01/docs_latest_release/docs/tpu-mlir/developer_manual/html/

## 3. 用户界面

### 3.1. 模型转换过程

基本操作过程是用 `model_transform.py` 将模型转成 mlir 文件，然后用 `model_deploy.py` 将 mlir 转成对应的 model。

以 `somenet.onnx` 模型为例：

```bash
# To MLIR
model_transform.py \
    --model_name somenet \
    --model_def  somenet.onnx \
    --test_input somenet_in.npz \
    --test_result somenet_top_outputs.npz \
    --mlir somenet.mlir

# To Float Model
model_deploy.py \
   --mlir somenet.mlir \
   --quantize F32 \ # F16/BF16
   --processor BM1684X \
   --test_input somenet_in_f32.npz \
   --test_reference somenet_top_outputs.npz \
   --model somenet_f32.bmodel
```

#### 3.1.1. 支持图片输入

```bash
model_transform.py \
    --model_name img_input_net \
    --model_def img_input_net.onnx \
    --input_shapes [[1,3,224,224]] \
    --mean 103.939,116.779,123.68 \
    --scale 1.0,1.0,1.0 \
    --pixel_format bgr \
    --test_input cat.jpg \
    --test_result img_input_net_top_outputs.npz \
    --mlir img_input_net.mlir
```

#### 3.1.2. 支持多输入

```bash
model_transform.py \
    --model_name multi_input_net \
    --model_def  multi_input_net.onnx \
    --test_input multi_input_net_in.npz \ # a.npy,b.npy,c.npy
    --test_result multi_input_net_top_outputs.npz \
    --mlir multi_input_net.mlir
```

#### 3.1.3. 支持 INT8 对称和非对称

```bash
run_calibration.py somenet.mlir \
    --dataset dataset \
    --input_num 100 \
    -o somenet_cali_table

model_deploy.py \
   --mlir somenet.mlir \
   --quantize INT8 \
   --calibration_table somenet_cali_table \
   --processor BM1684X \
   --test_input somenet_in_f32.npz \
   --test_reference somenet_top_outputs.npz \
   --tolerance 0.9,0.7 \
   --model somenet_int8.bmodel
```

#### 3.1.4. 支持混精度

```bash
run_calibration.py somenet.mlir \
    --dataset dataset \
    --input_num 100 \
    --inference_num 30 \
    --expected_cos 0.99 \
    --calibration_table somenet_cali_table \
    --processor BM1684X \
    --search search_qtable \
    --quantize_method_list KL,MSE \
    --quantize_table somenet_qtable

model_deploy.py \
   --mlir somenet.mlir \
   --quantize INT8 \
   --calibration_table somenet_cali_table \
   --quantize_table somenet_qtable \
   --processor BM1684X \
   --model somenet_mix.bmodel
```

#### 3.1.5. 支持量化模型 TFLite

```bash
model_transform.py \
    --model_name resnet50_tf \
    --model_def  ../resnet50_int8.tflite \
    --input_shapes [[1,3,224,224]] \
    --mean 103.939,116.779,123.68 \
    --scale 1.0,1.0,1.0 \
    --pixel_format bgr \
    --test_input ../image/dog.jpg \
    --test_result resnet50_tf_top_outputs.npz \
    --mlir resnet50_tf.mlir

model_deploy.py \
    --mlir resnet50_tf.mlir \
    --quantize INT8 \
    --processor BM1684X \
    --test_input resnet50_tf_in_f32.npz \
    --test_reference resnet50_tf_top_outputs.npz \
    --tolerance 0.95,0.85 \
    --model resnet50_tf_1684x.bmodel
```

#### 3.1.6. 支持 Caffe 模型

```bash
model_transform.py \
    --model_name resnet18_cf \
    --model_def  ../resnet18.prototxt \
    --model_data ../resnet18.caffemodel \
    --input_shapes [[1,3,224,224]] \
    --mean 104,117,123 \
    --scale 1.0,1.0,1.0 \
    --pixel_format bgr \
    --test_input ../image/dog.jpg \
    --test_result resnet50_cf_top_outputs.npz \
    --mlir resnet50_cf.mlir
```

#### 3.1.7. 支持 LLM 模型

```bash
llm_convert.py \
    -m /workspace/Qwen2.5-VL-3B-Instruct-AWQ \
    -s 2048 \
    -q w4bf16 \
    -c bm1684x \
    --max_pixels 672,896 \
    -o qwen2.5vl_3b
```

### 3.2. 工具参数介绍

#### 3.2.1. model_transform.py

将各种神经网络模型转换成 MLIR 文件（`.mlir` 后缀）以及配套的权重文件（`${model_name}_top_${quantize}_all_weight.npz`）

| 参数名 | 必选？ | 说明 |
|--------|--------|------|
| `--model_name` | 是 | 指定模型名称 |
| `--model_def` | 是 | 指定模型定义文件，如 `.onnx` / `.tflite` / `.prototxt` |
| `--mlir` | 是 | 指定输出的 mlir 文件名称和路径，`.mlir` 后缀 |
| `--input_shapes` | 否 | 指定输入的 shape，例如 `[[1,3,640,640]]` |
| `--model_extern` | 否 | 其他模型定义文件，用于合并 |
| `--model_data` | 否 | 指定模型权重文件，caffe 模型需要 |
| `--input_types` | 否 | 指定输入的类型，例如 int32 |
| `--keep_aspect_ratio` | 否 | resize 时是否保持长宽比，默认 false |
| `--mean` | 否 | 图像每个通道的均值，默认为 0.0,0.0,0.0 |
| `--scale` | 否 | 图片每个通道的比值，默认为 1.0,1.0,1.0 |
| `--pixel_format` | 否 | 图片类型：rgb/bgr/gray/rgbd，默认 bgr |
| `--channel_format` | 否 | 通道类型：nhwc/nchw/none，默认 nchw |
| `--output_names` | 否 | 指定输出的名称 |
| `--add_postprocess` | 否 | 将后处理融合到模型中 |
| `--test_input` | 否 | 指定输入文件用于验证 |
| `--test_result` | 否 | 指定验证后的输出文件，`.npz` 格式 |
| `--excepts` | 否 | 指定需要排除验证的网络层的名称 |
| `--onnx_sim` | 否 | onnx-sim 参数，目前仅支持 `skip_fuse_bn` |
| `--debug` | 否 | 保存可用于 debug 的模型 |
| `--tolerance` | 否 | 模型转换的余弦与欧式相似度的误差容忍度，默认 0.99,0.99 |
| `--cache_skip` | 否 | 是否跳过正确性检查 |
| `--dynamic_shape_input_names` | 否 | 具有动态 shape 的输入的名称列表 |
| `--dynamic` | 否 | 自动将 dynamic_axis 输入加入动态列表 |
| `--resize_dims` | 否 | 预处理前的原始输入图像尺寸 h,w |
| `--pad_value` | 否 | 图片缩放时边框填充大小 |
| `--pad_type` | 否 | 填充类型：normal/center |
| `--preprocess_list` | 否 | 输入是否需要做预处理的选项 |
| `--log_level` | 否 | 日志输出级别 |

转成 mlir 文件后，会生成一个 `${model_name}_in_f32.npz` 文件。

#### 3.2.2. run_calibration.py

用少量样本做 calibration，得到网络的校准表（每层 op 的 threshold/min/max）。

| 参数名 | 必选？ | 说明 |
|--------|--------|------|
| `<mlir>` | 是 | 指定 mlir 文件 |
| `--sq` | 否 | SmoothQuant |
| `--smc` | 否 | Softmax 修正 |
| `--we` | 否 | 跨层权重均衡 |
| `--bc` | 否 | 偏差校正 |
| `--dataset` | 否 | 指定输入样本的目录 |
| `--data_list` | 否 | 指定样本列表 |
| `--input_num` | 否 | 指定校准数量，0 为使用全部样本 |
| `--inference_num` | 否 | search 过程中所需推理图片数量 |
| `--tune_num` | 否 | 指定微调样本数量，默认 10 |
| `--histogram_bin_num` | 否 | 直方图 bin 数量，默认 2048 |
| `--expected_cos` | 否 | 期望 search_qtable 混精模型输出与浮点模型输出的相似度 |
| `--max_float_layers` | 否 | search_qtable 浮点层数量 |
| `--processor` | 否 | 处理器类型 |
| `--cali_method` | 否 | 选择量化门限计算方法 |
| `--fp_type` | 否 | search_qtable 浮点层数据类型 |
| `--search` | 否 | 搜索类型：search_qtable / search_threshold / false |
| `--transformer` | 否 | 是否是 transformer 模型 |
| `--quantize_method_list` | 否 | search_qtable 用来搜索的门限方法 |
| `--part_quantize` | 否 | 指定模型部分量化 |
| `--quantize_table` | 否 | search_qtable 输出的混精度量化表 |
| `-o` | 是 | 输出 calibration table 文件 |

校准表格式：

```
# op_name    threshold    min    max
images 1.0000080 0.0000000 1.0000080
122_Conv 56.4281803 -102.5830231 97.6811752
...
```

#### 3.2.3. model_deploy.py

将 mlir 文件转换成相应的 model。

| 参数名 | 必选？ | 说明 |
|--------|--------|------|
| `--mlir` | 是 | 指定 mlir 文件 |
| `--processor` | 是 | 平台：BM1684/BM1684X/BM1688/BM1690/CV186X/CV183X/CV182X/**CV181X**/CV180X |
| `--quantize` | 是 | 量化类型：F32/F16/**BF16**/INT8 等 |
| `--quant_input` | 否 | 输入数据类型是否与量化类型一致 |
| `--quant_output` | 否 | 输出数据类型是否与量化类型一致 |
| `--quantize_table` | 否 | 指定混精度量化表路径 |
| `--fuse_preprocess` | 否 | 是否将预处理融合到模型中 |
| `--calibration_table` | 否 | 指定校准表路径 |
| `--high_precision` | 否 | 打开时部分算子固定用 float32 |
| `--tolerance` | 否 | MLIR 量化后结果与 MLIR fp32 推理结果的容忍度，默认 0.8,0.5 |
| `--test_input` | 否 | 指定输入文件用于验证 |
| `--test_reference` | 否 | 验证模型正确性的参考数据 |
| `--op_divide` | 否 | **CV183x/CV182x/CV181x/CV180x only**，拆大 op 节省 ion 内存 |
| `--model` | 是 | 指定输出的 model 文件路径 |
| `--debug` | 否 | 是否保留中间文件 |
| `--asymmetric` | 否 | 指定做 int8 非对称量化 |
| `--skip_validation` | 否 | 跳过检查 bmodel 的正确性 |
| `--merge_weight` | 否 | 将权重与之前生成的 cvimodel 合并 |
| `--opt` | 否 | LayerGroup 优化类型：1/2/3，**默认 2**。1=简单模式(快)；2=动态编译找全局最优；3=线性规划(训练图) |
| `--disable_layer_group` | 否 | 是否关闭 LayerGroup |
| `--addr_mode` | 否 | 地址分配模式：auto/basic/io_alone/io_tag/io_reloc/in_reuse，默认 auto |
| `--layer_group_config` | 否 | 指定 layer group JSON 配置文件的路径 |
| `--shape_secs_search_strategy` | 否 | LayerGroup 中 shape_secs 搜索策略：0/1/2，默认 0，越大越优但编译越慢 |
| `--lgcache` | 否 | 是否暂存 LayerGroup 切分结果，默认 true |
| `--disable_gdma_check` | 否 | 是否关闭 gdma 地址检查 |
| `--compress_mode` | 否 | 压缩模式：none/weight/activation/all |

不同处理器支持的量化类型：

| 处理器 | 支持的 quantize |
|--------|----------------|
| BM1684 | F32, INT8 |
| BM1684X | F32, F16, BF16, INT8, W4F16, W8F16, W4BF16, W8BF16 |
| BM1688 | F32, F16, BF16, INT8, INT4, W4INT8, W4F16, ... |
| CV186X | F32, F16, BF16, INT8, INT4 |
| **CV183X/CV182X/CV181X/CV180X** | **BF16, INT8** |
| CV184X | BF16, INT8, W4INT8 |

#### 3.2.4. llm_convert.py

略。

#### 3.2.5. llm_analyse.py

| 参数名 | 必选？ | 说明 |
|--------|--------|------|
| `--chip` | 是 | 处理器类型 |
| `--tpu_freq` | 否 | 指定 TPU 频率，单位 MHz |

#### 3.2.6. model_runner.py

略。

#### 3.2.7. npz_tool.py

| 功能 | 描述 |
|------|------|
| dump | 得到 npz 的所有 tensor 信息 |
| compare | 比较 2 个 npz 文件的差异 |
| to_dat | 将 npz 导出为 dat 文件 |

#### 3.2.8. visual.py

量化精度对比可视化工具。

| 功能 | 描述 |
|------|------|
| `--f32_mlir` | fp32 网络 mlir 文件 |
| `--quant_mlir` | 量化后网络 mlir 文件 |
| `--input` | 测试输入数据 |
| `--port` | TCP 端口，默认 10000 |

#### 3.2.9. mlir2graph.py

基于 dot 对 mlir 文件可视化。

#### 3.2.10 gen_rand_input.py

生成随机输入数据。

#### 3.2.11 model_tool

处理 bmodel/cvimodel 的工具。

- `--info`：查看基本信息
- `--combine`：合并多个 bmodel
- `--extract`：分解 bmodel
- `--weight`：显示权重信息
- `--encrypt`/`--decrypt`：加解密

#### 3.2.12 mlir_cut

对 mlir 文件进行截断。

## 8. Lowering

Lowering 将 Top 层 OP 下沉到 Tpu 层 OP，支持 F32/F16/BF16/INT8 对称/INT8 非对称。

### 8.1. 基本过程

- Top 算子分 f32 和 int8 两种
- f32 算子可直接转成 f32/f16/bf16 的 tpu 层算子
- 要转 int8 则需要 `calibrated_type`
- int8 算子只能直接转成 tpu 层 int8 算子

### 8.2. 混合精度

当 OP 之间的类型不一致时，则插入 CastOp。输出类型与输入类型相同。

## 10. LayerGroup

### 10.1. 基本概念

- **GMEM**（片外内存）：大（如 4GB）
- **LMEM**（片内内存）：小（如 32KB=CV181X）

LayerGroup 让尽可能多的 OP 经过切分后在 Local Memory 执行，避免反复的 Local↔Global Memory 拷贝。

**基本思路**：通过切 Activation 的 N 和 H，使每层 Layer 的运算始终在 Local Memory 中。

### 10.2. BackwardH

对网络进行 H 切分时，Conv/Pool 等需要特别计算（反向计算输入 H 范围）。

### 10.3. 划分 Mem 周期

把每层 Layer 需要的 lmem 归为三类：
1. **Activation Tensor**：输入输出结果，无使用者后释放
2. **Weight**：权重，不切则用完释放；常驻则一直占用
3. **Buffer**：中间结果，用完释放

按广度优先配置 id，再配置周期（TimeStep）。

### 10.4. LMEM 分配

- 有 N/H 切分时：weight 常驻 LMEM
- 无切分时：weight 和 activation 一样处理
- 分配策略：按 op 顺序，优先分配 timestep 长的，次分配 lmem 大的

### 10.5. 划分最优 Group

从尾部向头部方向划分 group，优先切 N，N 切到最小还不够则切 H。

当 backward 后的 layer 的输入 H slice 重复部分 > H/2 时认为失败。

## 11. GMEM 分配

### 11.1. 目的

节约 global memory 空间，最大程度复用内存空间。

分配顺序：weight tensor → global neuron tensor（根据生命周期分配）

### 11.2. 原理

#### 11.2.1 weight tensor 分配

遍历所有 WeightOp，依次分配，4K 地址对齐。

#### 11.2.2 global neuron tensors 分配

根据生命周期分配，5 个条件才能 reuse 地址：
1. 不在 hold_edges 内
2. 地址不在 in_using_addr 内
3. 已 EOL（end of life）
4. 地址空间 >= 当前 tensor 所需空间
5. 当前 OP 的输入 tensor 地址不能与之相同

## 12. CodeGen

### 12.1. 主要工作

将 mlir 文件转换成最终的 bmodel 文件。

### 12.2. 工作流程

三部分：指令生成 → 指令存储 → 指令取出生成 Bmodel

数据结构：
- `bdc_buffer` / `gdma_buffer`：TIU/GDMA 指令存储
- `gdma_total_id` / `bdc_total_id`：指令总数目

### 12.3-12.5

后端函数通过动态库加载（`libbackend_xxx.so`），store_cmd 使用单例 + 装饰器模式管理指令存储。
