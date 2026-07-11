# 测试图像保存功能说明

## 功能概述

训练过程中，**每10个epoch会自动保存测试结果图像**，方便可视化模型训练进度和效果。

## 保存位置

测试图像保存在：
```
./pretrained/{model_name}/{quality_level}/test_images_epoch_{epoch}/
```

例如：
```
./pretrained/DMC_slf_yuv420_lsui/2/test_images_epoch_0/
./pretrained/DMC_slf_yuv420_lsui/2/test_images_epoch_10/
./pretrained/DMC_slf_yuv420_lsui/2/test_images_epoch_20/
...
```

## 保存内容

每个epoch文件夹包含：
- `{image_name}_input.png` - 输入的退化图像
- `{image_name}_gt.png` - Ground Truth（目标清晰图像）
- `{image_name}_output.png` - 模型输出的恢复图像

所有图像都自动从YCbCr色彩空间转换为RGB进行保存。

## 查看测试结果

### 1. 列出所有可用的epoch

```bash
python view_test_results.py --list
```

输出示例：
```
Available test results in ./pretrained/DMC_slf_yuv420_lsui/2:
--------------------------------------------------
Epoch   0: 1 test images
Epoch  10: 1 test images
Epoch  20: 1 test images
...
Total: 18 epoch(s) with test images
```

### 2. 查看特定epoch的结果

```bash
python view_test_results.py --epoch 10
```

这会：
- 显示该epoch的所有测试图像（input、GT、output）
- 自动生成对比图（包含三张图像的拼接）
- 保存对比图为 `{image_name}_comparison.png`

### 3. 指定模型和质量等级

```bash
python view_test_results.py \
    --model DMC_slf_yuv420_lsui \
    --quality_level 2 \
    --epoch 20
```

## 手动浏览

直接进入对应的文件夹查看PNG图像：

```bash
cd pretrained/DMC_slf_yuv420_lsui/2/test_images_epoch_10
ls -la
# 会看到：
# 1_input.png   - 输入图像
# 1_gt.png      - GT图像  
# 1_output.png  - 模型输出
```

## 修改保存频率

如果想改变保存频率（例如每5个epoch保存一次），修改 `train_vd_phase_3.py`:

```python
# 原来：每10个epoch保存
save_images = (epoch % 10 == 0)

# 改为：每5个epoch保存
save_images = (epoch % 5 == 0)
```

## 注意事项

1. **只保存第一个batch的第一帧**：为了节省空间，每个epoch只保存一个测试样本用于可视化
2. **只在主进程保存**：分布式训练时只有rank=0的进程保存图像，避免重复
3. **自动创建目录**：首次保存时会自动创建对应的epoch文件夹
4. **YCbCr→RGB转换**：所有保存的PNG图像都是RGB格式，便于直接查看

## 磁盘空间

假设每张图像约500KB，保存频率为每10个epoch：
- 180个epoch → 18次保存
- 每次3张图像（input、GT、output）
- 总空间约：18 × 3 × 0.5MB = 27MB

非常节省空间！

## 训练监控建议

1. **定期查看**：每隔几个epoch查看一次结果，观察模型是否在进步
2. **对比不同epoch**：对比epoch 0、10、20...的输出，可以直观看到训练效果
3. **发现问题**：如果输出图像出现异常（如颜色失真、伪影等），可以及早发现并调整

## 示例workflow

```bash
# 1. 启动训练
bash train_lsui.sh

# 2. 训练一段时间后，查看可用的测试结果
python view_test_results.py --list

# 3. 查看epoch 10的结果
python view_test_results.py --epoch 10

# 4. 查看epoch 20的结果
python view_test_results.py --epoch 20

# 5. 对比不同epoch，观察训练进度
```

这样就可以直观地监控模型训练效果了！🎨

