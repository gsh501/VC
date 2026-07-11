# TensorBoard 完整使用指南

## 快速开始

### 1. 启动训练
```bash
bash train_lsui.sh
```

### 2. 启动TensorBoard
```bash
# 新开一个终端
bash start_tensorboard.sh
```

### 3. 打开浏览器
```
http://localhost:6006
```

## TensorBoard记录内容

### 📊 SCALARS - 标量指标

#### 训练指标 (Train/) - 每500步更新
- **Train/Loss** - 训练损失值
- **Train/PSNR** - 训练PSNR（峰值信噪比）
- **Train/SSIM** - 训练SSIM（结构相似性）

#### 测试指标 (Test/) - 每个epoch更新

**基本指标：**
- **Test/Loss** - 测试集平均损失
- **Test/PSNR** - 测试集平均PSNR
- **Test/MSE** - 测试集平均MSE
- **Test/NumImages** - 测试的图像数量

**PSNR统计：**
- **Test/PSNR_Min** - 最低PSNR值
- **Test/PSNR_Max** - 最高PSNR值
- **Test/PSNR_Std** - PSNR标准差

**Loss统计：**
- **Test/Loss_Min** - 最低损失值
- **Test/Loss_Max** - 最高损失值
- **Test/Loss_Std** - 损失标准差

### 📊 DISTRIBUTIONS / HISTOGRAMS - 分布图

#### 测试集PSNR分布
- **Test/PSNR_Distribution** - 显示所有测试图像的PSNR分布
  - 可以看到PSNR的集中趋势
  - 识别异常值（特别好或特别差的图像）
  - 观察分布如何随epoch变化

#### 测试集Loss分布
- **Test/Loss_Distribution** - 显示所有测试图像的Loss分布
  - 了解模型在不同图像上的性能差异
  - 发现难以处理的图像类型

### 🖼️ IMAGES - 图像可视化（每10个epoch）

- **Test_Images/Input** - 输入的退化图像
- **Test_Images/GT** - Ground Truth清晰图像
- **Test_Images/Output** - 模型输出的恢复图像
- **Test_Images/Comparison** - 三张图像对比（并排显示）

## 如何使用TensorBoard监控训练

### 1. 查看训练进度

**SCALARS标签页：**
1. 勾选 `Train/Loss`, `Train/PSNR`, `Train/SSIM`
2. 观察曲线趋势：
   - ✅ Loss应该逐渐下降
   - ✅ PSNR应该逐渐上升
   - ✅ SSIM应该逐渐接近1

**调整平滑度：**
- 左侧调整"Smoothing"滑块（推荐0.6-0.8）
- 平滑后更容易看清趋势

### 2. 评估测试性能

**查看平均指标：**
1. 勾选 `Test/PSNR`, `Test/Loss`, `Test/MSE`
2. 观察是否稳定提升
3. 检查是否出现过拟合（训练好但测试不好）

**查看统计信息：**
1. 勾选 `Test/PSNR_Min`, `Test/PSNR_Max`, `Test/PSNR_Std`
2. 观察：
   - Min和Max的差距是否缩小（模型更稳定）
   - Std是否减小（性能更一致）

### 3. 分析PSNR分布

**DISTRIBUTIONS标签页：**
1. 查看 `Test/PSNR_Distribution`
2. 理解分布形状：
   - **正态分布**：大部分图像性能相似（好）
   - **长尾分布**：有些图像特别难处理（需要关注）
   - **多峰分布**：可能有不同类型的图像

**随epoch变化：**
- 使用滑块查看不同epoch的分布
- 理想情况：分布逐渐向右移动（PSNR提高）

### 4. 可视化检查

**IMAGES标签页：**
1. 查看 `Test_Images/Comparison`
2. 对比Input、GT、Output
3. 关注细节：
   - 颜色是否准确
   - 纹理是否清晰
   - 有无伪影

**时间对比：**
- 使用滑块查看不同epoch的输出
- 观察视觉质量的提升

## 实用技巧

### 技巧1: 对比多次训练

如果调整了超参数重新训练：
```bash
# TensorBoard会自动识别同一目录下的多次运行
tensorboard --logdir=./pretrained/DMC_slf_yuv420_lsui/2/tensorboard
```

在界面上勾选不同的运行进行对比。

### 技巧2: 识别训练问题

**Loss不下降：**
- 检查学习率是否过小
- 查看PSNR是否在提升（有时Loss不代表一切）

**训练不稳定（曲线波动大）：**
- 降低学习率
- 增加batch size
- 检查 `Test/PSNR_Std` 是否过大

**过拟合迹象：**
- `Train/PSNR` ↑ 但 `Test/PSNR` 停滞或 ↓
- `Test/PSNR_Std` 增大

### 技巧3: 找到最佳checkpoint

1. 在SCALARS中找到 `Test/PSNR` 最高的epoch
2. 对应的checkpoint保存在：
   ```
   ./pretrained/DMC_slf_yuv420_lsui/2/checkpoint_best_loss_vd.pth.tar
   ```

### 技巧4: 导出数据分析

1. 点击图表右上角的下载按钮
2. 选择"Download as CSV"
3. 在Excel或Python中进一步分析

## 示例分析流程

### 场景1: 检查训练是否正常

```
1. 打开 SCALARS → Train/Loss
   ✓ 如果持续下降 → 正常
   ✗ 如果不变或上升 → 检查学习率

2. 打开 SCALARS → Test/PSNR
   ✓ 如果逐渐上升 → 正常
   ✗ 如果停滞 → 可能需要更多训练时间

3. 打开 DISTRIBUTIONS → Test/PSNR_Distribution
   ✓ 如果分布向右移动 → 整体提升
   ✗ 如果分布变宽 → 某些图像性能下降
```

### 场景2: 对比不同训练配置

```
1. 运行训练1（learning_rate=1e-4）
2. 修改参数，运行训练2（learning_rate=5e-5）
3. TensorBoard中勾选两次运行
4. 对比：
   - 哪个收敛更快？
   - 哪个最终PSNR更高？
   - 哪个更稳定（Std更小）？
```

### 场景3: 诊断性能问题

```
如果 Test/PSNR 不理想：

1. 查看 Test/PSNR_Distribution
   - 是否有低PSNR的离群点？
   
2. 查看 Test_Images/Comparison
   - 哪种类型的图像恢复不好？
   
3. 查看 Test/PSNR_Std
   - 标准差大 → 模型对不同图像表现不一致
   - 标准差小 → 可能整体性能偏低

4. 根据发现调整策略：
   - 增加特定类型图像的训练数据
   - 调整数据增强策略
   - 修改模型架构
```

## 远程服务器使用

如果在远程服务器训练：

```bash
# 服务器上
bash start_tensorboard.sh

# 本地建立SSH隧道
ssh -L 6006:localhost:6006 username@server_ip

# 本地浏览器
http://localhost:6006
```

## 常见问题

**Q: 图表一开始是空的**
A: 等待训练开始记录数据（约500步后）

**Q: Distribution图看不懂**
A: 横轴是PSNR值，纵轴是该值出现的频率。颜色表示不同epoch。

**Q: 想清空旧数据重新开始**
A: 
```bash
rm -rf ./pretrained/DMC_slf_yuv420_lsui/2/tensorboard
bash train_lsui.sh  # 重新训练
```

**Q: TensorBoard占用内存太多**
A: 可以定期清理旧的运行记录，或者减少记录频率

## 性能说明

- **训练指标**: 每500步记录一次（不会影响训练速度）
- **测试指标**: 每个epoch记录一次
- **图像**: 每10个epoch保存一次（节省空间）
- **测试集**: 现在测试**所有图像**，可以看到完整的性能分布

## 总结

TensorBoard提供的信息：
- ✅ 实时训练曲线
- ✅ 每个epoch的测试性能
- ✅ 所有测试图像的PSNR/Loss分布
- ✅ PSNR的统计信息（Min/Max/Std）
- ✅ 视觉效果对比

充分利用这些信息可以更好地理解和优化模型训练！📊🎯

