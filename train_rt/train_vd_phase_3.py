import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from src.models.video_t import DMC
from src.models.image_model import DMCI
from compressai.datasets import ImageFolder  
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np

from src.dataload import DataSet,TetsDataSet
from src.dataload_lsui import LSUIDataSet, LSUITestDataSet
import torch.distributed as dist
import glob
import imageio
from src.utils.transforms import ycbcr2rgb


def adjust_learning_rate(optimizer, epoch, initial_lr, factors):
    """
    手动调整学习率，根据 epoch 和预设的衰减因子进行调整。
    
    参数:
        optimizer: 当前使用的优化器
        epoch: 当前的 epoch
        initial_lr: 初始学习率
        factors: 对应每个阶段的衰减因子
    """
    lr = initial_lr

    # 修复：根据180个epoch的训练周期调整学习率衰减点
    if epoch >= 150:
        lr *= factors[3]  
    elif epoch >= 120:
        lr *= factors[2]  
    elif epoch >= 80:
        lr *= factors[1]  
    elif epoch >= 40:
        lr *= factors[0]  

    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




class RateDistortionLoss(nn.Module):
    """自定义率失真损失函数（带拉格朗日参数）"""
    def __init__(self, lamada=3600):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self,epoch, result, target, lamada):
        N, _, H, W = target.size()
        out = {}
        # 只使用 MSE 和 SSIM 作为损失（忽略 bpp）
        out["mse_loss"] = result["mse"]
        # result["ssim"] 可能是 per-sample 值或张量，计算其均值作为 SSIM 指标
        ssim_val = result.get("ssim")
        if isinstance(ssim_val, torch.Tensor):
            ssim_mean = ssim_val.mean(dim=list(range(1, ssim_val.dim()))) if ssim_val.dim() > 1 else ssim_val
            ssim_mean = ssim_mean.mean()
        else:
            ssim_mean = float(ssim_val) if ssim_val is not None else 0.0
        out["ssim"] = ssim_mean

        # 综合损失：主要以 MSE 为主，并加入 (1 - SSIM) 项以鼓励感知质量
        # 修复：早期使用固定权重帮助收敛，后期使用lambda动态调整
        if epoch < 50:  # 前50个epoch使用固定权重
            out["loss"] = out["mse_loss"] * 100
        else:  # 50个epoch后使用lambda和SSIM项
            out["loss"] = lamada * out["mse_loss"] + (1.0 - out["ssim"]) * 0.01
       
        return out

class AverageMeter:
    """计算运行过程中的平均值"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """自定义 DataParallel 以便访问模型内的方法"""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def init(args):
    base_dir = f'./pretrained/{args.model}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    logging.info('Logging file is %s' % log_dir)

def Var(x):
    return Variable(x.cuda())

def calculate_psnr(x, x_hat, max_val=1.0):
    mse = F.mse_loss(x, x_hat, reduction='mean')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr

def psnr(x, x_hat, max_val=1.0):
    y_hat_420,uv_hat_420 = yuv_444_to_420(x_hat)
    y_420,uv_420,= yuv_444_to_420(x)
    u_420 = uv_420[:, 0:1, :, :]
    v_420 = uv_420[:, 1:2, :, :]
    u_hat_420 = uv_hat_420[:, 0:1, :, :]
    v_hat_420 = uv_hat_420[:, 1:2, :, :]
     
    psnr_y = calculate_psnr(y_420, y_hat_420, max_val)
    psnr_u = calculate_psnr(u_420, u_hat_420, max_val)
    psnr_v = calculate_psnr(v_420, v_hat_420, max_val)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8.0
    return psnr

def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    
    return ckpt

def get_sync_random_value(epoch, i):
    # 由主进程生成qs_global
    if epoch < 48:
        qs_global = 71
    else:
        
        if i % 3 == 0:
            qs_global = 71
        else:
            qs_global = random.randint(0, 70)

    return qs_global

def sync_random_value(qs_global):
    # 将qs_global的值广播到所有进程
    qs_global_tensor = torch.tensor(qs_global).cuda()
    dist.broadcast(qs_global_tensor, src=0)  # 广播到所有GPU
    return qs_global_tensor.item()



def qp_to_lambda(qp, q_num=72, lam_min=1, lam_max=768):
    """
    将整型QP映射到实数 lambda，QP 取值范围：[0, q_num - 1]
    """
    scale = qp / (q_num - 1)
    ln_lam_min = math.log(lam_min)
    ln_lam_max = math.log(lam_max)
    ln_lambda = ln_lam_min + scale * (ln_lam_max - ln_lam_min)
    return math.exp(ln_lambda)


index_map = [0, 1, 0, 2,0,1,0,2]
weights = [0.5,1.2,0.5,0.9, 0.5,1.2,0.5,0.9]
#############################
# 训练和测试函数定义（加入多帧多阶段训练策略）
#############################

def train_one_epoch(epoch, model, i_frame_net, criterion, train_dataloader, optimizer, gpu_per_batch, clip_max_norm, writer=None):
    """
    支持单帧与多帧训练：
      - 使用LSUI数据集，包含input和GT图像对
      - ref: GT的第一帧
      - input_images: 输入的退化图像序列
      - gt_images: GT图像序列（用于计算loss）
    """
    model.train()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    for i, d in enumerate(train_dataloader):
        # LSUI数据集返回: ref_image, input_images, gt_images
        ref_from_dataset, input_images, gt_images = Var(d[0]), Var(d[1]), Var(d[2])
        ref_from_dataset = ref_from_dataset.cuda(non_blocking=True)
        input_images = input_images.cuda(non_blocking=True)
        gt_images = gt_images.cuda(non_blocking=True)
        
        # 将输入和GT按帧拆分（每帧3通道）
        input_images = list(input_images.split(3, dim=1))
        gt_images = list(gt_images.split(3, dim=1))
        
        # 使用input的第一帧作为ref（而不是GT），与推理保持一致
        ref = input_images[0]  # 使用input的第一帧作为ref

        optimizer.zero_grad()

       # 在训练循环中
        qs_global = get_sync_random_value(epoch, i)  # 获取主进程生成的qs_global
        qs_global = sync_random_value(qs_global)  # 广播到所有进程


        lamada_qs = qp_to_lambda(qs_global)

        idx = 1
        ssim_list = []
        psnr_list = []
        total_loss = 0.0 
#      
  
        if qs_global>63:
            ref_out = i_frame_net.compress_(ref, 63)
        else:
            ref_out = i_frame_net.compress_(ref, qs_global)
   
        # Extract x_hat from the output dictionary
        ref_recon = ref_out["x_hat"]

        model.module.clear_dpb()
        model.module.add_ref_frame(None, ref_recon)

        # 使用input图像作为输入，GT图像计算loss
        for input_image, gt_image in zip(input_images, gt_images):
          
            current_input = input_image
            current_gt = gt_image
            # fa_idx = index_map[idx % 8]
            # curr_qp = model.module.shift_qp(qs_global, fa_idx)
            
            # 强制调用 forward_water（如果使用 DDP，调用 module.forward_water）
            # 传入 gt 参数，使 forward_water 使用 x_hat 和 gt 计算 MSE 和 SSIM
            if hasattr(model, 'module') and hasattr(model.module, 'forward_water'):
                out_net = model.module.forward_water(current_input, qs_global, gt=current_gt)
            elif hasattr(model, 'forward_water'):
                out_net = model.forward_water(current_input, qs_global, gt=current_gt)
            else:
                out_net = model(current_input, qs_global)

            if idx==1:
                lamada = 1.0*lamada_qs
            else:
                lamada = lamada_qs
                
            ssim_list.append(out_net["ssim"])
            # 使用GT图像计算PSNR
            psnr_list.append(psnr(current_gt, out_net["x_hat"]))

            # 使用GT图像计算loss
            out_criterion = criterion(epoch, out_net, current_gt, lamada*weights[idx % 8])
            loss_i = out_criterion["loss"].mean()
            total_loss += loss_i

            idx += 1

        total_loss.backward()
        if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        

        if i % 500 == 0:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            if dist.get_rank() == 0:
                logging.info(
                    f'[{i}/{len(train_dataloader.dataset)/gpu_per_batch/4}] | '
                    f'Multi-frame Loss: {total_loss.item():.3f} | '
                    f'PSNR: {avg_psnr:.3f} | '
                    f'SSIM: {avg_ssim.mean():.3f}'
                    )

                # TensorBoard logging
                if writer is not None:
                    global_step = epoch * len(train_dataloader) + i
                    writer.add_scalar('Train/Loss', total_loss.item(), global_step)
                    writer.add_scalar('Train/PSNR', avg_psnr, global_step)
                    writer.add_scalar('Train/SSIM', avg_ssim.mean(), global_step)


def tensor_to_image(tensor):
    """Convert tensor to numpy image for saving"""
    # tensor shape: [3, H, W] in YCbCr space
    tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
    rgb = ycbcr2rgb(tensor)  # Convert to RGB
    rgb = torch.clamp(rgb * 255, 0, 255)
    rgb = rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return rgb


def tensor_to_tensorboard(tensor):
    """Convert tensor to RGB tensor for TensorBoard [3, H, W] in range [0, 1]"""
    # Convert to RGB tensor for TensorBoard [3, H, W] in range [0, 1]
    tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
    rgb = ycbcr2rgb(tensor)  # Convert to RGB
    rgb = torch.clamp(rgb, 0, 1)
    return rgb.squeeze(0)  # [3, H, W]

def save_test_images(epoch, input_img, gt_img, output_img, base_dir, image_name):
    """Save test images to disk"""
    save_dir = os.path.join(base_dir, f"test_images_epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    
    input_rgb = tensor_to_image(input_img)
    gt_rgb = tensor_to_image(gt_img)
    output_rgb = tensor_to_image(output_img)
    
    # Save images to disk
    base_name = os.path.splitext(image_name)[0]
    imageio.imwrite(os.path.join(save_dir, f"{base_name}_input.png"), input_rgb)
    imageio.imwrite(os.path.join(save_dir, f"{base_name}_gt.png"), gt_rgb)
    imageio.imwrite(os.path.join(save_dir, f"{base_name}_output.png"), output_rgb)


def test_epoch(epoch, i_frame_net, test_dataloader, model, criterion, test_num, base_dir=None, save_images=False, writer=None):
    model.eval()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    # Store per-image results for TensorBoard
    per_image_psnr = []
    per_image_loss = []
    
    # Collect all test images for TensorBoard visualization
    all_input_images = []
    all_gt_images = []
    all_output_images = []
    all_image_names = []

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            # LSUI测试数据集返回: input_images, gt_images, image_names
            input_images, gt_images, image_names = d[0], d[1], d[2]
            input_images = Var(input_images)
            gt_images = Var(gt_images)
            input_images = input_images.cuda(non_blocking=True)
            gt_images = gt_images.cuda(non_blocking=True)
            
            # input_images和gt_images的shape: [B, N, 3, H, W]
            # 取第一帧input作为参考（而不是GT），与推理保持一致
            ref = input_images[:, 0, :, :, :]
            
            # 修复：与训练时保持一致，根据QP计算lambda，而不是硬编码
            qs_global = 71
            lamada_qs = qp_to_lambda(qs_global)
            
            # 这里假设 `d` 是一个包含多个帧的序列，长度由 test_num 确定
            total_loss = 0.0
            total_psnr = 0.0
            if qs_global>63:
                ref_out = i_frame_net.compress_(ref, 63)
            else:
                ref_out = i_frame_net.compress_(ref, qs_global)
            
            # Extract x_hat from the output dictionary
            ref_recon = ref_out["x_hat"]
            
            # 兼容 DDP 和单模型
            if hasattr(model, 'module'):
                model.module.clear_dpb()
                model.module.add_ref_frame(None, ref_recon)
            else:
                model.clear_dpb()
                model.add_ref_frame(None, ref_recon)
            
            # 从第1帧开始测试（第0帧是参考帧）
            for j in range(1, min(test_num + 1, input_images.size(1))):
                
                # 对每个帧进行处理
                current_input = input_images[:, j, :, :, :]
                current_gt = gt_images[:, j, :, :, :]
                # fa_idx = index_map[j % 8]
                # curr_qp = model.module.shift_qp(qs_global, fa_idx)
                
                # 强制调用 forward_water
                # 传入 gt 参数，使 forward_water 使用 x_hat 和 gt 计算 MSE 和 SSIM
                if hasattr(model, 'module') and hasattr(model.module, 'forward_water'):
                    out_net = model.module.forward_water(current_input, qs_global, gt=current_gt)
                elif hasattr(model, 'forward_water'):
                    out_net = model.forward_water(current_input, qs_global, gt=current_gt)
                else:
                    out_net = model(current_input, qs_global)

                # 修复：与训练时保持一致，第一帧使用1.0*lamada_qs，其他帧使用lamada_qs，并应用weights
                if j == 1:
                    lamada = 1.0 * lamada_qs
                else:
                    lamada = lamada_qs

                # 使用GT计算loss，与训练时保持一致（使用weights）
                out_criterion = criterion(epoch, out_net, current_gt, lamada*weights[j % 8])

                # 更新各个指标（不再使用 bpp）
                total_loss += out_criterion["loss"].mean()
                total_psnr += psnr(current_gt, out_net["x_hat"])
               
                # 收集所有测试图像用于 TensorBoard 可视化
                if dist.get_rank() == 0:
                    # 处理batch中的所有样本
                    batch_size = current_input.size(0)
                    for b in range(batch_size):
                        # 将tensor移动到CPU以便后续处理和保存
                        input_cpu = current_input[b].cpu()
                        gt_cpu = current_gt[b].cpu()
                        output_cpu = out_net["x_hat"][b].cpu()
                        
                        all_input_images.append(input_cpu)
                        all_gt_images.append(gt_cpu)
                        all_output_images.append(output_cpu)
                        # 为每帧生成唯一的图像名称
                        img_name = image_names[b] if isinstance(image_names[b], str) else str(image_names[b])
                        frame_name = f"{img_name}_frame{j}"
                        all_image_names.append(frame_name)
                        
                        # 保存图像到磁盘（如果需要）
                        if save_images and base_dir is not None:
                            save_test_images(
                                epoch,
                                input_cpu,  # [3, H, W] (CPU tensor)
                                gt_cpu,     # [3, H, W] (CPU tensor)
                                output_cpu,  # [3, H, W] (CPU tensor)
                                base_dir,
                                frame_name
                            )

            # 计算序列的平均损失和 PSNR
            avg_loss = total_loss / test_num
            avg_psnr = total_psnr / test_num

            # 更新指标的平均值
            loss_meter.update(avg_loss.item())
            psnr_meter.update(avg_psnr)
            mse_meter.update(out_criterion["mse_loss"].mean())
            
            # Store per-image results
            per_image_psnr.append(avg_psnr)
            per_image_loss.append(avg_loss.item())
    
    if dist.get_rank() == 0:
        logging.info(
            f"Test epoch {epoch}: Average Loss: {loss_meter.avg:.3f} | "
            f"Average PSNR: {psnr_meter.avg:.3f} | "
            f"Average MSE: {mse_meter.avg:.8f} | "
            f"Tested {len(per_image_psnr)} images | "
            f"Collected {len(all_input_images)} images for visualization\n"
        )
        
        # TensorBoard logging for test metrics
        if writer is not None:
            # Average metrics
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/PSNR', psnr_meter.avg, epoch)
            writer.add_scalar('Test/MSE', mse_meter.avg, epoch)
            writer.add_scalar('Test/NumImages', len(per_image_psnr), epoch)
            
            # Distribution of per-image PSNR and Loss
            if len(per_image_psnr) > 0:
                psnr_tensor = torch.tensor(per_image_psnr)
                loss_tensor = torch.tensor(per_image_loss)
                
                # Add histograms
                writer.add_histogram('Test/PSNR_Distribution', psnr_tensor, epoch)
                writer.add_histogram('Test/Loss_Distribution', loss_tensor, epoch)
                
                # Add statistics
                writer.add_scalar('Test/PSNR_Min', psnr_tensor.min().item(), epoch)
                writer.add_scalar('Test/PSNR_Max', psnr_tensor.max().item(), epoch)
                writer.add_scalar('Test/PSNR_Std', psnr_tensor.std().item(), epoch)
                writer.add_scalar('Test/Loss_Min', loss_tensor.min().item(), epoch)
                writer.add_scalar('Test/Loss_Max', loss_tensor.max().item(), epoch)
                writer.add_scalar('Test/Loss_Std', loss_tensor.std().item(), epoch)
            
            # Add all test images to TensorBoard with unique labels
            if len(all_input_images) > 0:
                for idx, (input_img, gt_img, output_img, img_name) in enumerate(zip(
                    all_input_images, all_gt_images, all_output_images, all_image_names
                )):
                    # Convert to RGB tensors for TensorBoard
                    input_tb = tensor_to_tensorboard(input_img)
                    gt_tb = tensor_to_tensorboard(gt_img)
                    output_tb = tensor_to_tensorboard(output_img)
                    
                    # Sanitize image name for TensorBoard tag (remove invalid characters)
                    safe_name = img_name.replace('/', '_').replace('\\', '_')
                    
                    # Add each image with unique tag
                    writer.add_image(f'Test_Images/Input/{safe_name}', input_tb, epoch)
                    writer.add_image(f'Test_Images/GT/{safe_name}', gt_tb, epoch)
                    writer.add_image(f'Test_Images/Output/{safe_name}', output_tb, epoch)
                    
                    # Also add a comparison grid for each image
                    comparison = torch.stack([input_tb, gt_tb, output_tb], dim=0)  # [3, 3, H, W]
                    writer.add_images(f'Test_Images/Comparison/{safe_name}', comparison, epoch)
    
    return loss_meter.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint_vd.pth.tar"):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(base_dir, filename),
                        os.path.join(base_dir, "checkpoint_best_loss_vd.pth.tar"))

#############################
# 参数解析
#############################
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training script for DMCI model with multi-frame multi-stage training on LSUI dataset.")
    parser.add_argument("-m", "--model", default="DMC_slf_yuv420_lsui", help="Model name (used for saving directory)")
    parser.add_argument("--dataset_root", type=str, default="/home/admin1/Data/water_enhance/LSUI", help="LSUI dataset root directory")
    parser.add_argument("-e", "--epochs", default=180, type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Number of dataloader threads")
    parser.add_argument("-q", "--quality-level", type=int, default=2, help="Quality level")
    parser.add_argument("--lamada", type=float, default=1024, help="Rate-distortion parameter")
    parser.add_argument("--batch-size", type=int, default=4, help="Initial batch size (for single-frame training)")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Patch size (default: %(default)s)")
    parser.add_argument("--frame-count", type=int, default=3, help="Number of frames per GOP (each image replicated N times)")
    parser.add_argument("--test-gop", type=int, default=3, help="GOP size for testing (each image replicated N times)")
    parser.add_argument('--local-rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument("--gpu-id", type=str, default=0, help="GPU id")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, help="Random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping max norm")
    parser.add_argument("--name", default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help="Result dir name")
    parser.add_argument("--model_path_i", type=str, default="checkpoints/cvpr2025_image.pth.tar", help="Path to I-frame model checkpoint")
    parser.add_argument("--checkpoint", type=str, default="pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar", help="Path to P-frame model checkpoint")
    parser.add_argument("--manyframe_epoch", type=int, default=120, help="Epoch threshold to switch to 7-frame training")
    args = parser.parse_args(argv)
    return args

#############################
# 主函数
#############################
def main(argv):
    args = parse_args(argv)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Initialize TensorBoard writer
    writer = None
    if dist.get_rank() == 0:
        tensorboard_dir = os.path.join(base_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        
        setup_logger(os.path.join(base_dir, time.strftime('%Y%m%d_%H%M%S') + '.log'))
        logging.info(f'======================= {args.name} =======================')
        for k, v in args.__dict__.items():
            logging.info(f'{k}: {v}')
        logging.info('=' * 40)
        logging.info(f'TensorBoard logging to: {tensorboard_dir}')

    # 使用LSUI数据集进行训练
    train_dataset = LSUIDataSet(
        root=args.dataset_root,
        im_height=args.patch_size[0],
        im_width=args.patch_size[1],
        frame_count=args.frame_count,
        train=True
    )
    
    # 使用LSUI测试数据集
    test_dataset = LSUITestDataSet(
        root=args.dataset_root,
        gop=args.test_gop,
        testfull=True,  # 测试所有图像以在TensorBoard中显示完整结果
        train=False
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.cuda and torch.cuda.is_available()),
        sampler=test_sampler
    )
    global_step = 0  # 全局训练步数
    gpu_per_batch = args.batch_size
    # 测试帧数为GOP size - 1（第一帧作为参考）
    test_num = args.test_gop - 1

    # 设置设备
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # 加载I帧模型
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net.eval()
    
    # 实例化P帧模型，并设置多GPU
    model = DMC()
    model = model.to(torch.device("cuda", args.local_rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # 冻结主干，只训练 recon_generation_net
    try:
        model.module.freeze_backbone_for_recon()
    except Exception:
        # 如果不是 DistributedDataParallel，直接调用
        try:
            model.freeze_backbone_for_recon()
        except Exception:
            pass

    # 仅将可训练参数传入优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    criterion = RateDistortionLoss(lamada=args.lamada)

    last_epoch = 0
    if args.checkpoint:
        if dist.get_rank() == 0:
           logging.info("Loading checkpoint from %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # 在加载 checkpoint 时将其记录的 epoch 重置为 0，训练从 epoch 0 开始
        if "epoch" in checkpoint:
            checkpoint["epoch"] = 0
        last_epoch = 0
        model.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    factors = [0.4, 0.1, 0.04, 0.01]  # 学习率衰减因子
    best_loss = float("inf")
    # 开始训练（按 epoch 循环）

    for epoch in range(last_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        if dist.get_rank() == 0:
            logging.info(f"====== Current epoch {epoch} ======")
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # 使用LSUI数据集，frame_count已在初始化时设置
        gpu_per_batch = args.batch_size
        test_num = args.test_gop - 1  # 测试帧数

        if dist.get_rank() == 0:
            logging.info(f"Training with {args.frame_count} frames per GOP")

        # 根据当前 gpu_per_batch 重构训练 DataLoader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=gpu_per_batch,
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=(args.cuda and torch.cuda.is_available())
        )
        
        # 训练一个 epoch（内部会根据 epoch 阶段控制帧数）
        
        train_one_epoch(epoch, model, i_frame_net, criterion, train_dataloader, optimizer, gpu_per_batch, args.clip_max_norm, writer=writer)
        
        # 每10个epoch保存测试图像
        save_images = (epoch % 10 == 0)
        loss = test_epoch(epoch, i_frame_net, test_dataloader, model, criterion, test_num, 
                         base_dir=base_dir, save_images=save_images, writer=writer)
        
        
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save :
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": None,
                },
                is_best,
                base_dir
            )

        # 更新 global_step（此处简单累加本 epoch 中迭代的 batch 数；实际中可根据精确训练步数更新）
        global_step += len(train_dataloader)
        if dist.get_rank() == 0:
           logging.info(f"Global step updated to: {global_step}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        if dist.get_rank() == 0:
            logging.info("TensorBoard writer closed")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


