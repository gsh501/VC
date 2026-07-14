#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：处理 MP4 视频
输入：单个 MP4 视频文件（通过 --input 指定，不支持文件夹或图像序列）
输出：处理后的视频帧（图像）或视频文件，以及平均 PSNR、平均 BPP、1s 传输总量

使用方法：
    # 使用 I帧 + P帧 模式（默认）
    python test_video_mp4.py --input video.mp4 --output_dir ./output --model_path_i checkpoints/cvpr2025_image.pth.tar --checkpoint pretrained/DMC_slf_yuv420_lsui/2/checkpoint_best_loss_vd.pth.tar

    # 更简洁的写法：直接把 mp4 路径作为第一个参数传入
    python test_video_mp4.py video.mp4 --output_dir ./output --model_path_i checkpoints/cvpr2025_image.pth.tar --checkpoint pretrained/DMC_slf_yuv420_lsui/2/checkpoint_best_loss_vd.pth.tar

    # 指定输出分辨率 + 采样帧率 + 处理时长（例：1080p 视频 → 512x256，每秒 10 帧，处理 2 秒，共 20 帧）
    python test_video_mp4.py --input video.mp4 --output_dir ./output --target_width 512 --target_height 256 --read_fps 10 --duration_sec 2 --model_path_i ... --checkpoint ...
    
    # 只使用 I帧模型（所有帧都用I帧模型处理，用于验证颜色一致性）
    python test_video_mp4.py --input video.mp4 --output_dir ./output_i --model_path_i checkpoints/cvpr2025_image.pth.tar --only_i_frame
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import cv2
import imageio.v2 as imageio
from tqdm import tqdm

# 导入模型和工具函数
from src.models.video_t import DMC
from src.models.image_model import DMCI
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, yuv_420_to_444, yuv_420_to_444


def get_state_dict(ckpt_path):
    """加载模型权重"""
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def rgb_to_ycbcr_tensor(rgb_frame):
    """
    将 RGB 帧转换为 YCbCr tensor（与训练时保持一致的处理流程）
    输入: RGB numpy array (H, W, 3) uint8
    输出: YCbCr tensor (3, H, W) float32 [0, 1]
    """
    # 转换为 (3, H, W) float32 [0, 1]
    rgb_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1).astype(np.float32) / 255.0)
    # 转换为 YCbCr 444
    rgb_tensor = rgb_tensor.unsqueeze(0)  # (1, 3, H, W)
    ycbcr_444 = rgb2ycbcr(rgb_tensor)
    # 转换为 420 格式再转回 444（与训练时保持一致，参考 dataload.py）
    ycbcr_420_y, ycbcr_420_uv = yuv_444_to_420(ycbcr_444)
    ycbcr_444 = yuv_420_to_444(ycbcr_420_y, ycbcr_420_uv)
    return ycbcr_444.squeeze(0)  # (3, H, W)


def ycbcr_tensor_to_rgb(ycbcr_tensor):
    """
    将 YCbCr tensor 转换为 RGB numpy array
    输入: YCbCr tensor (3, H, W) float32 [0, 1]
    输出: RGB numpy array (H, W, 3) uint8
    """
    ycbcr_tensor = ycbcr_tensor.unsqueeze(0)  # (1, 3, H, W)
    rgb_tensor = ycbcr2rgb(ycbcr_tensor)
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
    rgb_np = (
        rgb_tensor.squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    return rgb_np


def pad_to_multiple_of_64(tensor):
    """
    将 tensor 填充到 64 的倍数（模型要求）
    输入: tensor (3, H, W)
    输出: padded tensor (3, H', W'), (pad_h, pad_w)
    """
    _, h, w = tensor.shape
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return tensor, (pad_h, pad_w)


def unpad_tensor(tensor, pad_h, pad_w):
    """
    移除填充
    输入: tensor (3, H', W'), (pad_h, pad_w)
    输出: tensor (3, H, W)
    """
    if pad_h > 0 or pad_w > 0:
        _, h, w = tensor.shape
        tensor = tensor[:, :h-pad_h, :w-pad_w]
    return tensor


def compute_psnr(img1, img2, max_val=255.0):
    """
    计算两幅 RGB 图像之间的 PSNR (dB)。
    输入: img1, img2 为 (H, W, 3) uint8 或 [0,1] float 的 numpy 数组
    """
    if img1.dtype != np.float64 and img1.dtype != np.float32:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64 and img2.dtype != np.float32:
        img2 = img2.astype(np.float64)
    if img1.max() <= 1.0 and img2.max() <= 1.0:
        max_val = 1.0
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


def process_video(
    input_path,
    output_dir,
    i_frame_net,
    p_frame_net,
    device,
    qp=71,
    save_frames=True,
    save_video=True,
    target_width=None,
    target_height=None,
    output_fps=None,
    only_i_frame=False,
    read_fps=None,
    duration_sec=None,
):
    """
    处理视频。I 帧使用图像压缩模型，P 帧使用视频模型；DPB 由首帧（及后续 I 帧）重建初始化。

    参数:
        input_path: 输入视频路径
        output_dir: 输出目录
        i_frame_net: I帧模型（图像压缩）
        p_frame_net: P帧模型
        device: 设备
        qp: 量化参数
        save_frames: 是否保存帧图像
        save_video: 是否保存视频文件
        target_width: 处理分辨率宽度（None表示使用原始分辨率）
        target_height: 处理分辨率高度（None表示使用原始分辨率）
        output_fps: 输出视频FPS（None表示使用原始视频FPS）
        only_i_frame: 是否只使用I帧模型（所有帧都用I帧模型处理）
        read_fps: 采样帧率（每秒读取的帧数，如 10 表示每秒 10 帧）
        duration_sec: 处理时长（秒），与 read_fps 同时指定时生效；输入帧数 = read_fps * duration_sec
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    
    # 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算实际可读帧数（有些视频容器的元数据可能不准确）
    actual_frames = 0
    count_cap = cv2.VideoCapture(input_path)
    while True:
        ret, _ = count_cap.read()
        if not ret:
            break
        actual_frames += 1
    count_cap.release()
    cap.release()
    if actual_frames > 0:
        total_frames = actual_frames
    
    # 如果未指定分辨率，使用原始分辨率
    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height

    # 预处理：按 read_fps 和 duration_sec 决定读取哪些帧，并缩放到目标分辨率
    if read_fps is not None and duration_sec is not None:
        num_frames_to_process = int(read_fps * duration_sec)
        num_frames_to_process = max(1, min(num_frames_to_process, actual_frames))
        input_frames = []
        cap_pre = cv2.VideoCapture(input_path)
        current_idx = 0
        kept = 0
        while kept < num_frames_to_process:
            ret, frame = cap_pre.read()
            if not ret:
                break
            target_idx = round(kept * original_fps / read_fps) if original_fps and read_fps else kept
            if current_idx >= target_idx:
                if frame.shape[1] != target_width or frame.shape[0] != target_height:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_frames.append(frame)
                kept += 1
            current_idx += 1
        cap_pre.release()
        actual_frames = len(input_frames)
        total_frames = actual_frames
        print(f"预处理: 按 {read_fps} fps 采样 {duration_sec}s -> 共 {actual_frames} 帧, 分辨率 {target_width}x{target_height}")
    else:
        input_frames = []
        cap_pre = cv2.VideoCapture(input_path)
        while True:
            ret, frame = cap_pre.read()
            if not ret:
                break
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_frames.append(frame)
        cap_pre.release()
        actual_frames = len(input_frames)
        total_frames = actual_frames

    if not input_frames:
        raise ValueError("未读取到任何帧")

    # 输出 FPS：若指定了 duration_sec 则按「实际帧数/目标时长」计算，使输出视频时长严格等于处理时长（如处理 3s 则输出也是 3s）
    num_frames = len(input_frames)
    if output_fps is not None:
        output_fps = float(output_fps)
    elif duration_sec is not None and duration_sec > 0:
        output_fps = num_frames / float(duration_sec)
    elif read_fps is not None:
        output_fps = float(read_fps)
    else:
        output_fps = float(original_fps) if original_fps else 30.0
    output_fps = float(output_fps)
    
    # GOP 与 FPS 保持一致：若指定了 read_fps 则 GOP=read_fps，否则 GOP=output_fps（即约 1 秒一个 I 帧）
    gop = int(read_fps) if read_fps is not None else int(output_fps)
    gop = max(1, gop)

    print(f"视频信息: 原始 {width}x{height}, {original_fps:.2f} FPS; 本次处理 {actual_frames} 帧")
    print(f"处理分辨率: {target_width}x{target_height}")
    print(f"输出FPS: {output_fps:.2f}, GOP: {gop}")
    if only_i_frame:
        print(f"模式: 只使用I帧模型（所有帧都使用I帧模型处理）")
    else:
        print(f"模式: I帧 + P帧（每 {gop} 帧一个 I 帧，即第 0,{gop},{2*gop},... 帧为 I 帧）")
    
    # 准备视频写入器 - 使用cv2.VideoWriter，确保FPS设置正确
    video_writer = None
    if save_video:
        output_video_path = os.path.join(output_dir, 'output_video.mp4')
        # 尝试不同的编码器，按优先级顺序（mp4v通常在大多数系统上可用）
        codecs = ['mp4v', 'XVID', 'MJPG', 'avc1']
        video_writer = None
        used_codec = None
        
        for codec_name in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (target_width, target_height))
            if video_writer.isOpened():
                used_codec = codec_name
                break
        
        if video_writer is None or not video_writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器，尝试的所有编码器都失败: {codecs}")
        
        print(f"将保存视频到: {output_video_path}, FPS: {output_fps:.2f}, 编码器: {used_codec}")

    bpp_list = []
    psnr_list = []

    # 处理第一帧（I帧）
    frame = input_frames[0]
    frame_idx = 0
    print(f"处理第 {frame_idx + 1} 帧 (I帧)...")
    
    # 转换为 YCbCr
    frame_ycbcr = rgb_to_ycbcr_tensor(frame)
    frame_ycbcr, (pad_h, pad_w) = pad_to_multiple_of_64(frame_ycbcr)
    frame_ycbcr = frame_ycbcr.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # 使用 I帧模型处理
    with torch.no_grad():
        if qp > 63:
            ref_out = i_frame_net.compress_(frame_ycbcr, 63)
        else:
            ref_out = i_frame_net.compress_(frame_ycbcr, qp)
    
    if "bpp" in ref_out:
        bpp_list.append(ref_out["bpp"].item())
    
    ref_recon = ref_out["x_hat"]  # (1, 3, H, W) - 填充后的版本
    ref_recon_padded = ref_recon  # 保存填充后的版本用于 DPB
    
    # 转换为未填充版本用于保存
    ref_recon_unpadded = unpad_tensor(ref_recon.squeeze(0), pad_h, pad_w)  # (3, H, W)
    
    # 转换为 RGB 并保存
    output_frame = ycbcr_tensor_to_rgb(ref_recon_unpadded)
    psnr_list.append(compute_psnr(frame, output_frame))
    
    if save_frames:
        frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
        imageio.imwrite(frame_path, output_frame)
    
    if save_video:
        # cv2.VideoWriter需要BGR格式
        video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

    # DPB 使用首帧 I 帧的重建作为参考
    ref_recon_padded_for_dpb = ref_recon_padded

    # 如果只使用I帧模型，所有后续帧也使用I帧模型处理
    if only_i_frame:
        # 处理后续帧（全部使用I帧模型）
        with torch.no_grad():
            pbar = tqdm(total=max(len(input_frames) - 1, 0), desc="处理视频帧（I帧模式）")
            for frame_idx, frame in enumerate(input_frames[1:], start=1):
                pbar.update(1)
                pbar.set_description(f"处理第 {frame_idx + 1} 帧 (I帧模式)")
                frame_ycbcr = rgb_to_ycbcr_tensor(frame)
                frame_ycbcr, (pad_h, pad_w) = pad_to_multiple_of_64(frame_ycbcr)
                frame_ycbcr = frame_ycbcr.unsqueeze(0).to(device)  # (1, 3, H, W)
                if qp > 63:
                    out_net = i_frame_net.compress_(frame_ycbcr, 63)
                else:
                    out_net = i_frame_net.compress_(frame_ycbcr, qp)
                if "bpp" in out_net:
                    bpp_list.append(out_net["bpp"].item())
                output_frame_ycbcr = out_net["x_hat"]
                output_frame_ycbcr = unpad_tensor(output_frame_ycbcr.squeeze(0), pad_h, pad_w)
                output_frame = ycbcr_tensor_to_rgb(output_frame_ycbcr)
                psnr_list.append(compute_psnr(frame, output_frame))
                if save_frames:
                    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
                    imageio.imwrite(frame_path, output_frame)
                if save_video:
                    video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            pbar.close()
    else:
        # 初始化 P 帧模型的 DPB；后续按 GOP 在 I/P 之间切换（每 gop 帧一个 I 帧）
        p_frame_net.clear_dpb()
        p_frame_net.add_ref_frame(None, ref_recon_padded_for_dpb)
        with torch.no_grad():
            pbar = tqdm(total=max(len(input_frames) - 1, 0), desc="处理视频帧")
            for frame_idx, frame in enumerate(input_frames[1:], start=1):
                pbar.update(1)
                frame_ycbcr = rgb_to_ycbcr_tensor(frame)
                frame_ycbcr, (pad_h, pad_w) = pad_to_multiple_of_64(frame_ycbcr)
                frame_ycbcr = frame_ycbcr.unsqueeze(0).to(device)  # (1, 3, H, W)
                is_i_frame = (frame_idx % gop == 0)
                if is_i_frame:
                    pbar.set_description(f"处理第 {frame_idx + 1} 帧 (I帧)")
                    p_frame_net.clear_dpb()
                    if qp > 63:
                        out_net = i_frame_net.compress_(frame_ycbcr, 63)
                    else:
                        out_net = i_frame_net.compress_(frame_ycbcr, qp)
                    ref_recon_padded_new = out_net["x_hat"]
                    p_frame_net.add_ref_frame(None, ref_recon_padded_new)
                    output_frame_ycbcr = ref_recon_padded_new
                else:
                    pbar.set_description(f"处理第 {frame_idx + 1} 帧 (P帧)")
                    out_net = p_frame_net(frame_ycbcr, qp)
                    output_frame_ycbcr = out_net["x_hat"]
                if "bpp" in out_net:
                    bpp_list.append(out_net["bpp"].item())
                output_frame_ycbcr = unpad_tensor(output_frame_ycbcr.squeeze(0), pad_h, pad_w)
                output_frame = ycbcr_tensor_to_rgb(output_frame_ycbcr)
                psnr_list.append(compute_psnr(frame, output_frame))
                if save_frames:
                    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
                    imageio.imwrite(frame_path, output_frame)
                if save_video:
                    video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            pbar.close()

    if video_writer:
        video_writer.release()  # cv2.VideoWriter使用release()

    # 输出平均 PSNR 和平均 BPP
    pixel_per_frame = target_width * target_height
    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        print(f"\n平均 PSNR: {avg_psnr:.4f} dB (共 {len(psnr_list)} 帧)")
    if bpp_list:
        avg_bpp = np.mean(bpp_list)
        print(f"平均 BPP:  {avg_bpp:.6f} (共 {len(bpp_list)} 帧)")
        # 1 秒传输总量：1s 内所有帧的码流大小（与 fps 一致，如 10fps 即 1s 内 10 帧的 bin 总大小）
        bits_per_second = pixel_per_frame * avg_bpp * output_fps
        bytes_per_second = bits_per_second / 8.0
        print(f"1s 传输总量: {bits_per_second:.0f} bits = {bytes_per_second:.2f} B = {bytes_per_second/1024:.4f} KB (@ {output_fps:.1f} fps)")

    # 按 GOP 一个周期内各位置的平均值打印（如 GOP=10 则打印 10 个：I 平均值、第1个P 平均值、…）
    if bpp_list and psnr_list:
        n_frames = min(len(bpp_list), len(psnr_list))
        print(f"\n按 GOP 周期位置 (GOP={gop}, 各位置平均值):")
        for p in range(gop):
            indices = [i for i in range(p, n_frames, gop)]
            if not indices:
                continue
            bpp_p = np.mean([bpp_list[i] for i in indices])
            psnr_p = np.mean([psnr_list[i] for i in indices])
            bytes_p = (pixel_per_frame * bpp_p) / 8.0
            label = "I" if p == 0 else f"P{p}"
            print(f"  位置{p}({label}): 平均 bin = {bytes_p:.2f} B, 平均 PSNR = {psnr_p:.4f} dB (n={len(indices)})")
    
    print(f"\n处理完成！")
    print(f"输出目录: {output_dir}")
    if save_frames:
        print(f"帧图像保存在: {frames_dir}")
    if save_video:
        print(f"视频保存在: {output_video_path}")


def process_images(
    image_paths,
    output_dir,
    i_frame_net,
    p_frame_net,
    device,
    qp=71,
    save_frames=True,
    save_video=True,
    target_width=None,
    target_height=None,
    output_fps=None,
    only_i_frame=False,
):
    """
    处理图像序列（按顺序当作视频帧处理）。I 帧使用图像压缩，DPB 由首帧重建初始化。

    参数:
        image_paths: 图像路径列表
        output_dir: 输出目录
        i_frame_net: I帧模型（图像压缩）
        p_frame_net: P帧模型
        device: 设备
        qp: 量化参数
        save_frames: 是否保存帧图像
        save_video: 是否保存视频文件
        target_width: 处理分辨率宽度（None表示使用原始分辨率）
        target_height: 处理分辨率高度（None表示使用原始分辨率）
        output_fps: 输出视频FPS（None表示使用默认FPS=7）
        only_i_frame: 是否只使用I帧模型（所有帧都用I帧模型处理）
    """
    if not image_paths:
        raise ValueError("未提供图像路径")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    # 读取第一张图像以确定尺寸
    first_img = imageio.imread(image_paths[0])
    if first_img.ndim == 2:
        first_img = np.stack([first_img] * 3, axis=-1)
    if first_img.shape[-1] == 4:
        first_img = first_img[..., :3]
    if first_img.dtype != np.uint8:
        first_img = np.clip(first_img * 255.0, 0, 255).astype(np.uint8)

    height, width = first_img.shape[0], first_img.shape[1]
    total_frames = len(image_paths)

    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height

    if output_fps is None:
        output_fps = 7.0
    output_fps = float(output_fps)

    print(f"图像序列信息: {width}x{height}, {total_frames} 帧")
    print(f"处理分辨率: {target_width}x{target_height}")
    print(f"输出FPS: {output_fps:.2f}")
    if only_i_frame:
        print("模式: 只使用I帧模型（所有帧都使用I帧模型处理）")
    else:
        print("模式: I帧 + P帧（第0帧使用I帧，后续帧使用P帧）")

    # 准备视频写入器 - 使用cv2.VideoWriter，确保FPS设置正确
    video_writer = None
    if save_video:
        output_video_path = os.path.join(output_dir, 'output_video.mp4')
        codecs = ['mp4v', 'XVID', 'MJPG', 'avc1']
        video_writer = None
        used_codec = None
        for codec_name in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (target_width, target_height))
            if video_writer.isOpened():
                used_codec = codec_name
                break
        if video_writer is None or not video_writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器，尝试的所有编码器都失败: {codecs}")
        print(f"将保存视频到: {output_video_path}, FPS: {output_fps:.2f}, 编码器: {used_codec}")

    # 处理第一帧（I帧）
    frame = first_img
    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    frame_ycbcr = rgb_to_ycbcr_tensor(frame)
    frame_ycbcr, (pad_h, pad_w) = pad_to_multiple_of_64(frame_ycbcr)
    frame_ycbcr = frame_ycbcr.unsqueeze(0).to(device)

    with torch.no_grad():
        if qp > 63:
            ref_out = i_frame_net.compress_(frame_ycbcr, 63)
        else:
            ref_out = i_frame_net.compress_(frame_ycbcr, qp)

    ref_recon = ref_out["x_hat"]
    ref_recon_padded = ref_recon
    ref_recon_unpadded = unpad_tensor(ref_recon.squeeze(0), pad_h, pad_w)
    output_frame = ycbcr_tensor_to_rgb(ref_recon_unpadded)

    if save_frames:
        frame_path = os.path.join(frames_dir, f'frame_{0:06d}.png')
        imageio.imwrite(frame_path, output_frame)
    if save_video:
        video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

    # DPB 使用首帧 I 帧的重建作为参考
    ref_recon_padded_for_dpb = ref_recon_padded

    # 处理后续帧
    if only_i_frame:
        with torch.no_grad():
            pbar = tqdm(total=max(total_frames - 1, 0), desc="处理图像序列（I帧模式）")
            for idx, img_path in enumerate(image_paths[1:], start=1):
                pbar.update(1)
                pbar.set_description(f"处理第 {idx + 1} 帧 (I帧模式)")
                frame = imageio.imread(img_path)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                if frame.shape[1] != target_width or frame.shape[0] != target_height:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                frame_ycbcr = rgb_to_ycbcr_tensor(frame)
                frame_ycbcr, (pad_h, pad_w) = pad_to_multiple_of_64(frame_ycbcr)
                frame_ycbcr = frame_ycbcr.unsqueeze(0).to(device)
                if qp > 63:
                    out_net = i_frame_net.compress_(frame_ycbcr, 63)
                else:
                    out_net = i_frame_net.compress_(frame_ycbcr, qp)
                output_frame_ycbcr = out_net["x_hat"]
                output_frame_ycbcr = unpad_tensor(output_frame_ycbcr.squeeze(0), pad_h, pad_w)
                output_frame = ycbcr_tensor_to_rgb(output_frame_ycbcr)
                if save_frames:
                    frame_path = os.path.join(frames_dir, f'frame_{idx:06d}.png')
                    imageio.imwrite(frame_path, output_frame)
                if save_video:
                    video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            pbar.close()
    else:
        p_frame_net.clear_dpb()
        p_frame_net.add_ref_frame(None, ref_recon_padded_for_dpb)
        with torch.no_grad():
            pbar = tqdm(total=max(total_frames - 1, 0), desc="处理图像序列")
            for idx, img_path in enumerate(image_paths[1:], start=1):
                pbar.update(1)
                pbar.set_description(f"处理第 {idx + 1} 帧")
                frame = imageio.imread(img_path)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                if frame.shape[1] != target_width or frame.shape[0] != target_height:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                frame_ycbcr = rgb_to_ycbcr_tensor(frame)
                frame_ycbcr, (pad_h, pad_w) = pad_to_multiple_of_64(frame_ycbcr)
                frame_ycbcr = frame_ycbcr.unsqueeze(0).to(device)
                out_net = p_frame_net(frame_ycbcr, qp)
                output_frame_ycbcr = out_net["x_hat"]
                output_frame_ycbcr = unpad_tensor(output_frame_ycbcr.squeeze(0), pad_h, pad_w)
                output_frame = ycbcr_tensor_to_rgb(output_frame_ycbcr)
                if save_frames:
                    frame_path = os.path.join(frames_dir, f'frame_{idx:06d}.png')
                    imageio.imwrite(frame_path, output_frame)
                if save_video:
                    video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            pbar.close()

    if video_writer:
        video_writer.release()

    print("\n处理完成！")
    print(f"输出目录: {output_dir}")
    if save_frames:
        print(f"帧图像保存在: {frames_dir}")
    if save_video:
        print(f"视频保存在: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="处理 MP4 视频")
    parser.add_argument("input_video", nargs="?", default="data/2.mp4",
                       help="输入 MP4 视频文件路径；可直接写成第一个参数，例如 python test_video_mp4.py data/3.mp4")
    parser.add_argument("--input", dest="input_flag", type=str, default=None,
                       help="输入 MP4 视频文件路径（兼容旧写法）")
    parser.add_argument("--output_dir", type=str, default="video3", help="输出目录")
    parser.add_argument("--model_path_i", type=str, default="checkpoints/cvpr2025_image.pth.tar", 
                       help="I帧模型路径")
    parser.add_argument("--checkpoint", type=str, 
                       default="pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar",
                       help="P帧模型路径")
    parser.add_argument("--qp", type=int, default=15, help="量化参数 (0-71)")
    parser.add_argument("--save_frames", action="store_true", default=True, 
                       help="保存处理后的帧图像")
    parser.add_argument("--save_video", action="store_true", default=True,
                       help="保存处理后的视频文件")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--target_width", type=int, default=256, 
                       help="处理分辨率宽度（默认使用原始分辨率）")
    parser.add_argument("--target_height", type=int, default=128, 
                       help="处理分辨率高度（默认使用原始分辨率）")
    parser.add_argument("--output_fps", type=float, default=None,
                       help="输出视频FPS；不指定则与 duration_sec/read_fps 一致，保证输出时长=处理时长")
    parser.add_argument("--only_i_frame", action="store_true", default=False,
                       help="只使用I帧模型处理所有帧（用于验证I帧模型颜色一致性）")
    parser.add_argument("--read_fps", type=float, default=10,
                       help="采样帧率：每秒读取的帧数，如 10 表示每秒 10 帧；需与 --duration_sec 同时使用")
    parser.add_argument("--duration_sec", type=float, default=2.0,
                       help="处理时长（秒），与 --read_fps 同时使用；输入帧数 = read_fps * duration_sec，如 10fps 处理 2s 共 20 帧")
    
    args = parser.parse_args()

    input_path = args.input_flag or args.input_video or "data/1.mp4"
    input_path = os.path.expanduser(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入视频不存在: {input_path}")
    if not os.path.isfile(input_path):
        raise ValueError(f"输入路径不是文件: {input_path}")
    if not input_path.lower().endswith(".mp4"):
        raise ValueError(f"当前仅支持 MP4 文件，收到: {input_path}")

    print(f"输入视频: {input_path}")
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载 I帧模型
    print(f"加载 I帧模型: {args.model_path_i}")
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    
    # 只在需要P帧模型时加载
    p_frame_net = None
    if not args.only_i_frame:
        print(f"加载 P帧模型: {args.checkpoint}")
        p_frame_net = DMC()
        p_state_dict = get_state_dict(args.checkpoint)
        p_frame_net.load_state_dict(p_state_dict)
        p_frame_net = p_frame_net.to(device)
        p_frame_net.eval()
    else:
        print("跳过 P帧模型加载（只使用I帧模型）")
    
    process_video(
        input_path,
        args.output_dir,
        i_frame_net,
        p_frame_net,
        device,
        qp=args.qp,
        save_frames=args.save_frames,
        save_video=args.save_video,
        target_width=args.target_width,
        target_height=args.target_height,
        output_fps=args.output_fps,
        only_i_frame=args.only_i_frame,
        read_fps=args.read_fps,
        duration_sec=args.duration_sec,
    )


if __name__ == "__main__":
    main()
