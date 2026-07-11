import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def ycbcr420_to_444_np(y, uv, order=0, separate=False):
    '''
    y is 1xhxw Y float numpy array
    uv is 2x(h/2)x(w/2) UV float numpy array
    order: 0 nearest neighbor (default), 1: binear
    return value is 3xhxw YCbCr float numpy array
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def rgb2ycbcr(rgb, is_bgr=False):
    if is_bgr:
        b, g, r = rgb.chunk(3, -3)
    else:
        r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clamp(ycbcr, 0., 1.)
    return ycbcr


def ycbcr2rgb(ycbcr, is_bgr=False, clamp=True):
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if is_bgr:
        rgb = torch.cat((b, g, r), dim=-3)
    else:
        rgb = torch.cat((r, g, b), dim=-3)
    if clamp:
        rgb = torch.clamp(rgb, 0., 1.)
    return rgb


def yuv_444_to_420(yuv):
    def _downsample(tensor):
        return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    y = yuv[:, :1, :, :]
    uv = yuv[:, 1:, :, :]

    return y, _downsample(uv)

def yuv_420_to_444(y, uv_420):
    """
    将 YUV 4:2:0 恢复为 YUV 4:4:4。
    - y: Tensor, 形状 (B, 1, H, W)，Y 分量已经是全分辨率；
    - uv_420: Tensor, 形状 (B, 2, H/2, W/2)，UV 分量是 4:2:0 下采样后的半分辨率。
    
    返回:
    - yuv_444: Tensor, 形状 (B, 3, H, W)，通道顺序为 (Y, U, V)。
    """
    # 对 UV 通道进行上采样，放大到 (H, W)
    # 这里选用最近邻插值（mode='nearest'），以保证不引入额外的插值混叠
    uv_upsampled = F.interpolate(uv_420, scale_factor=2, mode='nearest')

    # 拼接回 Y 通道
    yuv_444 = torch.cat([y, uv_upsampled], dim=1)
    return yuv_444


def rgb2ycbcr444_np(rgb):
    """
    将RGB图像转换为YCbCr 4:4:4格式 (numpy版本)
    - rgb: numpy array, 形状 (3, H, W) uint8 [0, 255]
    - 返回: numpy array, 形状 (3, H, W) float [0, 1]，通道顺序为 (Y, Cb, Cr)
    """
    rgb = rgb.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    
    # ITU-R BT.709 权重
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # 计算Y, Cb, Cr
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    
    # 拼接为 (3, H, W)
    ycbcr = np.stack([y, cb, cr], axis=0)
    ycbcr = np.clip(ycbcr, 0.0, 1.0)
    
    return ycbcr

