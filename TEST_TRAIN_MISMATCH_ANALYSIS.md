# 训练/测试不一致问题分析

## 问题现象
- **训练曲线正常**：Train/Loss, Train/PSNR, Train/SSIM 在最后阶段都有恢复
- **测试曲线退化**：Test/Loss持续上升，Test/PSNR持续下降

## 关键差异

### 1. Lambda计算不一致 ⚠️ **严重**

**训练时（train_one_epoch）：**
```python
qs_global = get_sync_random_value(epoch, i)  # 随机QP或固定71
lamada_qs = qp_to_lambda(qs_global)  # 根据QP动态计算lambda

if idx==1:
    lamada = 1.0*lamada_qs
else:
    lamada = lamada_qs

out_criterion = criterion(epoch, out_net, current_gt, lamada*weights[idx % 8])
```

**测试时（test_epoch）：**
```python
qs_global = 71  # 固定QP
lamada = 768  # ❌ 硬编码的固定值，不是根据QP计算的！

out_criterion = criterion(epoch, out_net, current_gt, lamada)  # ❌ 没有weights
```

**问题**：
- 测试时lambda硬编码为768，应该用`qp_to_lambda(71)`计算
- 测试时没有使用`weights[idx % 8]`，训练时有

### 2. 权重使用不一致 ⚠️

**训练时：**
```python
lamada*weights[idx % 8]  # 使用了weights
```

**测试时：**
```python
lamada  # 没有使用weights
```

### 3. QP值策略

**训练时：**
- epoch < 48: 固定QP=71
- epoch >= 48: 随机QP (0-70) 或固定71（每3个batch固定一次71）

**测试时：**
- 始终固定QP=71

## 修复建议

### 修复1: 统一Lambda计算

测试时应该根据QP计算lambda，而不是硬编码：

```python
# 测试时
qs_global = 71
lamada_qs = qp_to_lambda(qs_global)  # 根据QP=71计算lambda
lamada = lamada_qs  # 使用计算出的lambda
```

### 修复2: 统一权重使用（如果需要）

如果训练时使用了weights，测试时也应该保持一致：

```python
# 测试时
for j in range(1, min(test_num + 1, input_images.size(1))):
    if j == 1:
        lamada = 1.0 * lamada_qs
    else:
        lamada = lamada_qs
    
    out_criterion = criterion(epoch, out_net, current_gt, lamada*weights[j % 8])
```

## 影响分析

训练和测试时使用不同的loss计算方式会导致：
1. **模型在训练集上学到的参数，在测试集上表现不一致**
2. **测试时loss计算方式与训练时不同，导致测试指标无法准确反映模型性能**
3. **这解释了为什么训练曲线恢复，但测试曲线持续退化** - 模型在训练集上过拟合了，但测试时的loss计算方式不同，导致性能表现差

