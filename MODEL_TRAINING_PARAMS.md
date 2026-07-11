# 模型训练参数说明

## 参数冻结策略

在 `train_vd_phase_3.py` 中，我们采用了**部分参数冻结**的策略，只训练重建网络部分。

### 冻结方法

```python
model.freeze_backbone_for_recon()
```

这个方法定义在 `src/models/video_t.py` 中。

## 冻结的参数（不训练）

以下模块的参数在训练时**被冻结**（`requires_grad = False`）：

### 1. **Feature Adaptor** (`feature_adaptor_i`, `feature_adaptor_p`)
- 功能：从参考帧提取特征
- 参数量：约3-4M

### 2. **Feature Extractor** (`feature_extractor`)
- 功能：提取特征并生成上下文
- 包含：
  - `conv1`: 两个DepthConvBlock
  - `conv2`: 四个DepthConvBlock
- 参数量：约5-6M

### 3. **Encoder** (`encoder`)
- 功能：编码输入图像
- 包含：
  - `conv1`: 1x1卷积
  - `conv2`: 两个DepthConvBlock
  - `conv3`: 一个DepthConvBlock
  - `down`: 下采样卷积
- 参数量：约2-3M

### 4. **Hyper Encoder** (`hyper_encoder`)
- 功能：编码超参数信息
- 参数量：约1-2M

### 5. **Hyper Decoder** (`hyper_decoder`)
- 功能：解码超参数信息
- 参数量：约1-2M

### 6. **Prior Fusion** (`y_prior_fusion`)
- 功能：融合先验信息
- 参数量：约1M

### 7. **Spatial Prior** (`y_spatial_prior`)
- 功能：空间先验
- 参数量：约1M

### 8. **Decoder** (`decoder`)
- 功能：解码特征
- 包含：
  - `up`: 上采样
  - `conv1`: DepthConvBlock
  - `conv2`: 两个DepthConvBlock
- 参数量：约2-3M

### 9. **量化参数** (除了 `q_recon`)
- `q_encoder`: 编码器量化参数
- `q_decoder`: 解码器量化参数
- `q_feature`: 特征量化参数

**总冻结参数量：约 17,226,000 个参数**

## 可训练的参数（训练）

以下模块的参数**可以训练**（`requires_grad = True`）：

### 1. **Recon Generation Net** (`recon_generation_net`)

**结构：**
```python
ReconGeneration(
  (conv): Sequential(
    (0): DepthConvBlock(256 -> 320)  # g_ch_d -> g_ch_recon
    (1): DepthConvBlock(320 -> 320)
    (2): DepthConvBlock(320 -> 320)
    (3): DepthConvBlock(320 -> 320)
  )
  (head): Conv2d(320 -> 192, kernel_size=1)  # g_ch_recon -> g_ch_src_d (3*8*8)
)
```

**功能：**
- 输入：从decoder输出的特征 (256维)
- 输出：重建的YCbCr图像 (3通道)
- 过程：
  1. 4个DepthConvBlock提升特征维度并细化
  2. 1x1卷积生成 192维输出 (3×8×8)
  3. PixelShuffle上采样8倍恢复图像尺寸
  4. Clamp到[0,1]范围

**参数量：约 3,400,000 个参数**

**详细分解：**
- DepthConvBlock (256->320): ~200K
- DepthConvBlock (320->320): ~250K × 3 = 750K
- Conv2d (320->192): ~61K
- 总计：~3.4M

### 2. **q_recon**

**类型：** `nn.Parameter`

**形状：** `[72+extra_qp, 320, 1, 1]`
- 72: QP数量
- extra_qp: 额外QP数量（8）
- 320: 通道数（g_ch_recon）

**功能：**
- 为不同QP提供量化步长
- 控制重建质量

**参数量：约 25,600 个参数 (80 × 320)**

## 训练参数统计

根据训练日志：

```
Freeze backbone: trainable params 3465472/20691456
```

- **可训练参数**: 3,465,472 (16.7%)
- **冻结参数**: 17,225,984 (83.3%)
- **总参数**: 20,691,456

## 为什么这样设计？

### 1. **保留预训练的压缩能力**

冻结编码器、解码器和特征提取器，保留模型在视频压缩任务上的能力：
- 编码效率不变
- 特征提取质量不变
- 比特率控制精确

### 2. **只微调重建质量**

`recon_generation_net` 是最后一步，负责从特征生成图像：
- 这是最影响视觉质量的部分
- 适合针对水下图像增强进行微调
- 不会破坏压缩能力

### 3. **减少训练成本**

只训练16.7%的参数：
- 训练速度更快
- 显存占用更少
- 更容易收敛

### 4. **避免灾难性遗忘**

冻结大部分参数：
- 保留原有的压缩知识
- 不会因为新数据集而忘记原有能力
- 模型在通用场景下仍然有效

## 训练流程

### Phase 1-2: 预训练（已完成）
- 训练整个模型
- 在大规模视频压缩数据集上
- 学习编码、解码、特征提取

### Phase 3: 水下图像微调（当前）
```python
# 1. 加载预训练模型
model.load_state_dict(checkpoint)

# 2. 冻结大部分参数
model.freeze_backbone_for_recon()

# 3. 只训练 recon_generation_net
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

# 4. 在LSUI数据集上微调
# Input: 水下退化图像
# Target: 清晰增强图像
```

## 验证参数冻结状态

在训练开始时，会打印：

```
Freeze backbone: trainable params 3465472/20691456
```

如果看到这行日志，说明参数冻结成功。

## 查看具体哪些参数可训练

可以在Python中运行：

```python
model = DMC()
model.freeze_backbone_for_recon()

# 查看所有参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"✓ Trainable: {name:50s} {param.numel():>10d}")
    else:
        print(f"✗ Frozen:    {name:50s} {param.numel():>10d}")
```

## 模块对应关系

| 模块 | 功能 | 状态 | 参数量 |
|------|------|------|--------|
| `feature_adaptor_i` | I帧特征适配 | ❄️ 冻结 | ~3M |
| `feature_adaptor_p` | P帧特征适配 | ❄️ 冻结 | ~1M |
| `feature_extractor` | 特征提取 | ❄️ 冻结 | ~5M |
| `encoder` | 图像编码 | ❄️ 冻结 | ~3M |
| `hyper_encoder` | 超参数编码 | ❄️ 冻结 | ~1M |
| `hyper_decoder` | 超参数解码 | ❄️ 冻结 | ~1M |
| `y_prior_fusion` | 先验融合 | ❄️ 冻结 | ~1M |
| `y_spatial_prior` | 空间先验 | ❄️ 冻结 | ~1M |
| `decoder` | 特征解码 | ❄️ 冻结 | ~2M |
| **`recon_generation_net`** | **图像重建** | ✅ **训练** | **~3.4M** |
| `q_encoder` | 编码量化参数 | ❄️ 冻结 | ~23K |
| `q_decoder` | 解码量化参数 | ❄️ 冻结 | ~23K |
| `q_feature` | 特征量化参数 | ❄️ 冻结 | ~23K |
| **`q_recon`** | **重建量化参数** | ✅ **训练** | **~26K** |

## 总结

- 🎯 **训练目标**: 只优化图像重建质量
- 🔒 **冻结83.3%参数**: 保留压缩能力
- ✅ **训练16.7%参数**: 适配水下图像增强
- 💡 **策略**: 迁移学习 + 参数高效微调

这种设计平衡了：
1. 保留原有能力
2. 适应新任务
3. 训练效率
4. 收敛稳定性

