# LSUI 数据集训练说明

## 数据集结构

LSUI数据集包含单张图像对（input和GT），而非视频序列：

```
LSUI/
├── input/          # 输入图像（退化/水下图像）
├── GT/             # Ground Truth图像（清晰/增强图像）
├── test_input/     # 测试集输入图像
└── test_gt/        # 测试集GT图像
```

## 数据加载策略

由于LSUI包含单张图像而非视频序列，我们采用**图像复制**策略来形成GOP（Group of Pictures）：

- 每张图像被复制3次形成一个3帧的GOP
- 训练时：ref = GT第一帧，input_images = 复制的输入图像，gt_images = 复制的GT图像
- 测试时：同样复制3次进行评估

## 文件说明

### 1. 数据加载器 (`src/dataload_lsui.py`)

包含两个类：

#### `LSUIDataSet` - 训练数据集
- 加载单张input/GT图像对
- 将每张图像复制`frame_count`次（默认3次）
- 支持随机裁剪和翻转数据增强
- 输出：
  - `ref_image`: [3, H, W] - GT参考帧
  - `input_images`: [frame_count*3, H, W] - 复制的输入帧
  - `gt_images`: [frame_count*3, H, W] - 复制的GT帧

#### `LSUITestDataSet` - 测试数据集
- 加载单张input/GT图像对
- 将每张图像复制`gop`次（默认3次）
- 自动裁剪到偶数尺寸以兼容YUV 4:2:0
- 输出：
  - `input_images`: [gop, 3, H, W]
  - `gt_images`: [gop, 3, H, W]
  - `image_name`: 文件名

### 2. 训练脚本 (`train_vd_phase_3.py`)

已修改以支持LSUI数据集：

**主要修改：**
- 使用`LSUIDataSet`和`LSUITestDataSet`
- 训练时使用input图像作为模型输入，GT图像计算loss和PSNR
- 支持GOP大小配置（默认3帧）

**关键参数：**
- `--dataset_root`: LSUI数据集根目录
- `--frame-count`: 每个GOP的帧数（默认3）
- `--test-gop`: 测试时的GOP大小（默认3）
- `--batch-size`: 批大小（默认4）
- `--patch-size`: 裁剪尺寸（默认256x256）

### 3. 启动脚本 (`train_lsui.sh`)

提供便捷的分布式训练启动方式。

### 4. 测试脚本 (`test_lsui_dataloader.py`)

用于验证数据加载器是否正常工作。

## 使用方法

### 1. 测试数据加载器

```bash
conda activate dcvc
python test_lsui_dataloader.py
```

### 2. 开始训练

#### 方法1：使用启动脚本（推荐）

```bash
conda activate dcvc
bash train_lsui.sh
```

#### 方法2：直接运行

```bash
conda activate dcvc

# 单GPU训练
python train_vd_phase_3.py \
    --dataset_root /home/admin1/Data/water_enhance/LSUI \
    --batch-size 4 \
    --frame-count 3 \
    --test-gop 3 \
    --epochs 180

# 多GPU分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_vd_phase_3.py \
    --dataset_root /home/admin1/Data/water_enhance/LSUI \
    --batch-size 4 \
    --frame-count 3 \
    --test-gop 3 \
    --epochs 180
```

## 训练流程

1. **数据加载**：加载单张input/GT图像对，复制3次
2. **I帧初始化**：使用I帧模型处理GT第一帧作为参考
3. **P帧训练**：
   - 输入：复制的input图像
   - 目标：复制的GT图像
   - 计算loss和PSNR时使用GT图像

## 关键特性

1. **单图像复制**：每张图像复制N次形成GOP
2. **Input/GT分离**：模型处理退化图像，使用清晰图像计算loss
3. **数据增强**：训练时支持随机裁剪和翻转
4. **偶数尺寸**：自动裁剪到偶数尺寸以兼容YUV 4:2:0
5. **灵活配置**：GOP大小、批大小、裁剪尺寸可配置

## 输出目录

训练模型保存在：
```
./pretrained/DMC_slf_yuv420_lsui/{quality_level}/
├── checkpoint_vd.pth.tar           # 最新checkpoint
├── checkpoint_best_loss_vd.pth.tar # 最佳checkpoint
└── {timestamp}.log                 # 训练日志
```

## 注意事项

1. **环境**：确保使用`dcvc` conda环境
2. **GPU数量**：根据可用GPU数量调整启动脚本中的`NUM_GPUS`
3. **批大小**：根据GPU显存调整`--batch-size`
4. **GOP大小**：固定为3（每张图像复制3次）
5. **图像尺寸**：测试时自动裁剪到偶数尺寸

## 测试结果示例

```
LSUI Dataset (train): Found 100 image pairs, each will be replicated 3 times
Dataset size: 100
Reference image shape: torch.Size([3, 256, 256])
Input images shape: torch.Size([9, 256, 256])  # 3 frames * 3 channels
GT images shape: torch.Size([9, 256, 256])     # 3 frames * 3 channels
```

## 故障排除

### 问题1：图像尺寸不匹配
**原因**：图像尺寸不是偶数，YUV 4:2:0转换失败
**解决**：数据加载器已自动裁剪到偶数尺寸

### 问题2：显存不足
**解决**：减小`--batch-size`或`--patch-size`

### 问题3：数据集路径错误
**解决**：检查`--dataset_root`是否指向正确的LSUI目录

