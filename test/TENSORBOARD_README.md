# TensorBoard 可视化说明

## 功能概述

训练过程中自动记录所有指标和图像到TensorBoard，方便实时监控训练进度和效果。

## TensorBoard日志位置

```
./pretrained/{model_name}/{quality_level}/tensorboard/
```

例如：
```
./pretrained/DMC_slf_yuv420_lsui/2/tensorboard/
```

## 启动TensorBoard

### 方法1: 使用启动脚本（推荐）

```bash
bash start_tensorboard.sh
```

然后在浏览器中打开: `http://localhost:6006`

### 方法2: 手动启动

```bash
tensorboard --logdir=./pretrained/DMC_slf_yuv420_lsui/2/tensorboard --port=6006 --bind_all
```

### 方法3: 指定不同端口

如果6006端口被占用：

```bash
tensorboard --logdir=./pretrained/DMC_slf_yuv420_lsui/2/tensorboard --port=6007 --bind_all
```

## 记录内容

### 1. 训练指标 (Train/)

**实时记录（每500个batch）：**
- `Train/Loss` - 训练损失
- `Train/PSNR` - 训练PSNR
- `Train/SSIM` - 训练SSIM

**图表说明：**
- X轴：Global step（全局训练步数）
- Y轴：指标值
- 可以看到训练过程中的波动和趋势

### 2. 测试指标 (Test/)

**每个epoch记录一次：**
- `Test/Loss` - 测试损失
- `Test/PSNR` - 测试PSNR
- `Test/MSE` - 测试MSE

**图表说明：**
- X轴：Epoch
- Y轴：指标值
- 显示模型在测试集上的性能变化

### 3. 测试图像 (Test_Images/)

**每10个epoch记录一次：**
- `Test_Images/Input` - 输入的退化图像
- `Test_Images/GT` - Ground Truth清晰图像
- `Test_Images/Output` - 模型输出的恢复图像
- `Test_Images/Comparison` - 三张图像的对比（并排显示）

**图像说明：**
- 自动从YCbCr转换为RGB
- 可以直观看到模型在不同epoch的恢复效果
- 可以使用滑块查看不同epoch的结果

## TensorBoard 界面使用

### SCALARS 标签页

**查看训练曲线：**
1. 左侧勾选要查看的指标
2. 使用平滑滑块调整曲线平滑度
3. 可以下载图表或数据

**对比不同运行：**
- 如果有多次训练，可以在左侧勾选不同的运行
- TensorBoard会自动用不同颜色显示

### IMAGES 标签页

**查看测试图像：**
1. 选择 `Test_Images/Input`、`Test_Images/GT`、`Test_Images/Output`
2. 使用滑块浏览不同epoch的结果
3. 点击图像可以放大查看
4. `Test_Images/Comparison` 显示三张图的对比

**快捷操作：**
- 鼠标滚轮：缩放图像
- 拖拽：移动图像
- 双击：重置视图

## 训练监控工作流

### 1. 启动训练

```bash
# 终端1: 启动训练
bash train_lsui.sh
```

### 2. 启动TensorBoard

```bash
# 终端2: 启动TensorBoard
bash start_tensorboard.sh
```

### 3. 实时监控

在浏览器中打开 `http://localhost:6006`，可以看到：

**训练阶段（每500步更新）：**
- 观察 `Train/Loss` 是否下降
- 观察 `Train/PSNR` 是否上升
- 观察 `Train/SSIM` 是否上升

**测试阶段（每个epoch更新）：**
- 观察 `Test/Loss` 是否下降
- 观察 `Test/PSNR` 是否上升
- 查看 `Test_Images` 的视觉效果

### 4. 问题诊断

**如果Loss不下降：**
- 检查学习率是否过大或过小
- 查看PSNR和SSIM是否有改善
- 检查测试图像是否有视觉改善

**如果出现过拟合：**
- `Train/PSNR` 持续上升，但 `Test/PSNR` 停滞或下降
- 考虑增加数据增强或减小模型容量

**如果训练不稳定：**
- Loss曲线剧烈波动
- 考虑降低学习率或增加batch size

## 高级使用

### 对比多次训练

如果你运行了多次训练（例如调整超参数），TensorBoard会自动识别：

```bash
# 训练1
bash train_lsui.sh

# 训练2（修改参数后）
bash train_lsui.sh

# TensorBoard会显示两次训练的曲线
tensorboard --logdir=./pretrained/DMC_slf_yuv420_lsui/2/tensorboard --port=6006
```

### 自定义平滑度

在TensorBoard界面左侧调整"Smoothing"滑块：
- 0: 显示原始数据
- 0.9: 高度平滑
- 推荐: 0.6-0.8

### 导出数据

1. 点击图表右上角的下载图标
2. 选择"Download as CSV"或"Download as SVG"
3. 可以在Excel或其他工具中进一步分析

### 远程访问

如果在远程服务器上训练：

```bash
# 在服务器上启动TensorBoard
tensorboard --logdir=./pretrained/DMC_slf_yuv420_lsui/2/tensorboard --port=6006 --bind_all

# 在本地建立SSH隧道
ssh -L 6006:localhost:6006 user@server
```

然后在本地浏览器访问 `http://localhost:6006`

## 常见问题

### Q: TensorBoard显示"No dashboards are active"

**A:** 等待一下，训练刚开始时可能还没有数据。或者检查日志目录是否正确。

### Q: 图像不显示

**A:** 确保训练已经运行了至少10个epoch（每10个epoch才保存图像）。

### Q: 曲线很乱看不清

**A:** 调整左侧的"Smoothing"滑块，增加平滑度。

### Q: 端口6006被占用

**A:** 使用其他端口：
```bash
tensorboard --logdir=... --port=6007
```

### Q: 想要清除旧的训练记录

**A:** 删除tensorboard目录：
```bash
rm -rf ./pretrained/DMC_slf_yuv420_lsui/2/tensorboard
```
下次训练会重新创建。

## 性能提示

1. **TensorBoard占用内存**: 如果训练很长时间，日志文件会变大，TensorBoard可能占用较多内存
2. **定期清理**: 可以定期删除不需要的旧日志
3. **采样率**: 当前设置每500步记录一次训练指标，已经足够密集

## 示例输出

训练开始时会看到：
```
2026-01-09 17:20:19,123 [INFO]  TensorBoard logging to: ./pretrained/DMC_slf_yuv420_lsui/2/tensorboard
```

这表示TensorBoard日志已经开始记录！🎯

