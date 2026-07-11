#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a *single* YUV420-p video sequence with DMCI (I-frame) + DMC (P-frame)
models, using a consistent 6 : 1 : 1 weighted PSNR metric identical to
`get_distortion()`.

2025-06-04
~~~~~~~~~~
* **`psnr()`** now takes **`x_hat, y, u, v`** (matching `get_distortion`).
* Dataset retains original Y and UV planes to avoid recomputing in Test.
* Helper `calc_psnr()` added for NumPy arrays.
* 在测试时同时打印 **Y-, U- 和 V-PSNR** 值，以及加权总体 PSNR。

Author: ChatGPT（修正版：直接使用原始 Y, UV 数据 + PSNR 通道输出）
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# ===== 工程依赖 =====
from src.utils.transforms import (
    ycbcr420_to_444_np,
    yuv_444_to_420,
)
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.video_reader import YUV420Reader
from src.models.image_model import DMCI
from src.models.video_t import DMC
# ====================

# ────────────────────────── 通用工具 ──────────────────────────

def np2tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """(3,H,W) np.uint8 → (1,3,H,W) float32 [0,1]"""
    return (
        torch.from_numpy(arr)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .div_(255.0)
    )




# ---------------------------------------------------------------------
# PSNR identical to get_distortion:   x_hat, y, u, v  → weighted PSNR
# ---------------------------------------------------------------------

def psnr(x_hat: torch.Tensor, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Compute 6:1:1 weighted PSNR given reconstructed tensor and GT planes."""
    # Convert reconstruction to YUV420 planes
    y_rec, uv_rec = yuv_444_to_420(x_hat)
    y_rec = (
        torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()[0]
    )  # (H,W)
    uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
    u_rec, v_rec = uv_rec[0], uv_rec[1]
    y = y.squeeze(0).cpu().numpy()[0]
    u = u.squeeze(0).cpu().numpy()[0]
    v = v.squeeze(0).cpu().numpy()[0]
    # Channel PSNRs
    
    psnr_y = calc_psnr(y, y_rec)
    psnr_u = calc_psnr(u, u_rec)
    psnr_v = calc_psnr(v, v_rec)

    # Weighted overall
    return (6 * psnr_y + psnr_u + psnr_v) / 8.0, psnr_y, psnr_u, psnr_v


def Var(x):
    return Variable(x.cuda() if torch.cuda.is_available() else x)


class RateDistortionLoss(nn.Module):
    """λ · MSE  +  bpp"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, result, target, lamada):
        out = {
            "bpp_loss": result["bpp"],
            "mse_loss": result["mse"],
        }
        out["loss"] = lamada * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def get_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt.get("net", ckpt))
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


# ──────────────────────── 单文件 Dataset ────────────────────────
class SingleYuvDataset(Dataset):
    def __init__(
        self,
        yuv_path: str,
        width: int,
        height: int,
        device: str = "cpu",
        max_frames: int = -1,
    ):
        self.yuv_path, self.device = yuv_path, device
        reader = YUV420Reader(yuv_path, width, height)

        frames = []  # 存储 YUV444 Tensor
        y_list = []  # 存储 原始 Y 平面 (H,W)
        uv_list = []  # 存储 原始 UV 平面 (2,H/2,W/2)

        while True:
            if max_frames > 0 and len(frames) >= max_frames:
                break
            got = reader.read_one_frame()
            if got is None:
                break
            y, uv = got  # Y:(H,W)  UV:(2,H/2,W/2)
            y_list.append(y)
            uv_list.append(uv)
            yuv444 = ycbcr420_to_444_np(y, uv)
            frames.append(np2tensor(yuv444, device))
        reader.close()

        if len(frames) < 2:
            raise RuntimeError(
                f"YUV 文件至少包含 2 帧（1×I + ≥1×P），但只读到 {len(frames)} 帧。"
            )
        self.frames = torch.cat(frames, dim=0)  # (N,3,H,W)
        self.y_list = y_list   # len=N, each (H,W)
        self.uv_list = uv_list  # len=N, each (2,H/2,W/2)
        self.name = os.path.basename(yuv_path)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        gop = self.frames.unsqueeze(0)  # (1,N,3,H,W)
        return gop, self.y_list, self.uv_list, self.name
    
index_map = [0, 1, 0, 2, 0, 1, 0, 2]

# ───────────────────────────── 测试循环 ─────────────────────────────
def Test(epoch, model_i, test_dataloader, model, criterion, test_num,QP):
    model.eval()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "psnr", "mse", "y", "u", "v")}

    with torch.no_grad():
        for gop, y_list, uv_list, name in test_dataloader:
            gop = gop.squeeze(1)
            gop = Var(gop)
            lamada, qs_global = 3600, QP # λ, QP
            # I-frame

            # P-frames
            last_qp = 63
            totals = {k: 0.0 for k in ("loss", "bpp", "psnr", "y", "u", "v")}
            for j in range(0, test_num):
                cur = gop[:, j].to(torch.float16)
                if j == 0:
                    if qs_global > 63:
                        out = model_i.compress_(cur, 63)
                    else:
                        out = model_i.compress_(cur, qs_global)

                    # P-frames
                    model.clear_dpb()
                    model.add_ref_frame(None, out["x_hat"])
                else:
                    if j%32== 1:
                        model.prepare_feature_adaptor_i(last_qp)

                    fa_idx = index_map[j % 8]
                    
                    curr_qp = model.shift_qp(qs_global, fa_idx)
                    last_qp = curr_qp
                    if QP > 63:
                      curr_qp = QP
                    out = model(cur, curr_qp)
                

                out_loss = criterion(out, cur, lamada)
                
                y_gt_np = y_list[j]
                uv_gt = uv_list[j]
                u_gt_np = uv_gt[:, 0:1, :, :]
                v_gt_np = uv_gt[:, 1:2, :, :]
          
                weighted_psnr, psnr_y, psnr_u, psnr_v = psnr(out["x_hat"], y_gt_np, u_gt_np, v_gt_np)

                # 打印每帧结果
                print(
                    f"Frame {j:02d}: Weighted PSNR: {weighted_psnr:.3f} dB | "
                    f"Y: {psnr_y:.3f} | U: {psnr_u:.3f} | V: {psnr_v:.3f} | "
                    f"Bpp: {out['bpp'].mean().item():.4f}"
                )

                totals["loss"] += out_loss["loss"].mean().item()
                totals["bpp"] += out["bpp"].mean().item()
                totals["psnr"] += weighted_psnr
                totals["y"] += psnr_y
                totals["u"] += psnr_u
                totals["v"] += psnr_v

            # Update meters (averaging over P-frames)
            for k in ("loss", "bpp", "psnr", "y", "u", "v"):
                meters[k].update(totals[k] / test_num)
            meters["mse"].update(out_loss["mse_loss"].mean().item())

            print(
                f"[{name[0]}] Weighted PSNR: {meters['psnr'].avg:.3f} dB | "
                f"Y-PSNR: {meters['y'].avg:.3f} dB | "
                f"U-PSNR: {meters['u'].avg:.3f} dB | "
                f"V-PSNR: {meters['v'].avg:.3f} dB | "
                f"MSE: {meters['mse'].avg:.5f} | "
                f"Bpp: {meters['bpp'].avg:.4f}\n"
            )



# ────────────────────────────── 主入口 ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single YUV420 video sequence with DMCI/DMC")
    parser.add_argument("--yuv",     default="/home/admin1/Data/water_date_test/test_data_yuv/1280_720/009.yuv", help=".yuv file path")
    parser.add_argument("--width",   type=int, default=1280, help="video width")
    parser.add_argument("--height",  type=int, default=720, help="video height")
    parser.add_argument("--ckpt_i",  default="checkpoints/cvpr2025_image.pth.tar", help="DMCI checkpoint")
    # parser.add_argument("--ckpt_p",  default="pretrained/DMC/3/checkpoint_vd.pth.tar", help="DMC  checkpoint")
    parser.add_argument("--ckpt_p",  default="pretrained/DMC_slf_yuv420_lsui/2/checkpoint_vd.pth.tar", help="DMC  checkpoint")

    parser.add_argument("--QP",  default=65, help="QS Global")
    parser.add_argument("--max_frames", type=int, default=10,
                        help="要测试的最大帧数（包括 I 帧）。传 -1 或不传则读取全部帧。")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args() 
    # checkpoints/cvpr2025_video.pth.tar
    # pretrained/DMCI/3/checkpoint_best_loss_vd.pth.tar
    # checkpoints/rt_v/25.6/3/checkpoint_best_loss_vd.pth.tar
    torch.backends.cudnn.benchmark = True   
    device = args.device

    dataset = SingleYuvDataset(
        args.yuv, args.width, args.height, device, args.max_frames
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    i_net = DMCI().to(device).half()
    i_net.load_state_dict(get_state_dict(args.ckpt_i))
    i_net.eval()

    p_net = DMC().to(device).half()
    p_net.load_state_dict(get_state_dict(args.ckpt_p))
    p_net.eval()
    QP = int(args.QP)
    criterion = RateDistortionLoss()

    total_frames = dataset.frames.size(0)
    use_frames = (
        min(total_frames, args.max_frames)
        if args.max_frames > 0
        else total_frames
    )
    test_num = use_frames 
    if test_num < 1:
        raise RuntimeError(
            f"读到的帧数只有 {use_frames}，不足以形成 1×I + ≥1×P 帧。"
        )

    Test(0, i_net, loader, p_net, criterion, test_num,QP)


if __name__ == "__main__":
    main()