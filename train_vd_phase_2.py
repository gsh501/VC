import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.video_t import DMC
from src.models.image_model import DMCI
from compressai.datasets import ImageFolder  
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np

from src.dataload import DataSet,TetsDataSet
import torch.distributed as dist


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


    if epoch >= 175:
        lr *= factors[3]  
    elif epoch >= 170:
        lr *= factors[2]  
    elif epoch >= 160:
        lr *= factors[1]  
    elif epoch >= 100:
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
        # bpp loss
        out["bpp_loss"] = result["bpp"]
        # MSE loss
        out["mse_loss"] = result["mse"]
        # 综合损失
        if 0 <= epoch < 20:
            out["loss"] = out["mse_loss"]*100000
        else:
            out["loss"] = lamada * out["mse_loss"]  + out["bpp_loss"]
       
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

def train_one_epoch(epoch,model,i_frame_net, criterion, train_dataloader, optimizer,gpu_per_batch,  clip_max_norm):
    """
    支持单帧与多帧训练：
      - 当 epoch < threeframe_epoch 时，使用单帧训练（原有逻辑）；
      - 当 threeframe_epoch <= epoch < fiveframe_epoch 时，使用 3 帧训练；
      - 当 epoch >= fiveframe_epoch 时，使用 5 帧训练。
    多帧训练时，假设输入数据为 5 维 [N, frame_num, C, H, W]，
    若数据集只返回 4 维图像，则自动复制扩充。
    """
    model.train()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    for i, d in enumerate(train_dataloader):
        # 判断输入数据维度；如果 frame_num > 1 而 d 只有 4 维，则扩充复制

        ref, input_images = Var(d[0]), Var(d[1])
        ref = ref.cuda(non_blocking=True)
        input_images = input_images.cuda(non_blocking=True)
        input_images = list(input_images.split(3, dim=1))
        

        optimizer.zero_grad()

       # 在训练循环中
        qs_global = get_sync_random_value(epoch, i)  # 获取主进程生成的qs_global
        qs_global = sync_random_value(qs_global)  # 广播到所有进程


        lamada_qs = qp_to_lambda(qs_global)

        idx = 1
        ssim_list = []
        psnr_list = []
        bpp_list = []
        total_loss = 0.0 
#       
  
        if qs_global>63:
            ref = i_frame_net.compress_(ref, 63)
        else:
            ref = i_frame_net.compress_(ref, qs_global)
   

        model.module.clear_dpb()
        model.module.add_ref_frame(None, ref)

        for input_image in input_images:
          
            current = input_image
            # fa_idx = index_map[idx % 8]
            # curr_qp = model.module.shift_qp(qs_global, fa_idx)
            
            out_net = model(current,  qs_global)

            if idx==1:
                lamada = 1.0*lamada_qs
            else:
                lamada = lamada_qs
                
            ssim_list.append(out_net["ssim"])
            psnr_list.append( psnr(current, out_net["x_hat"]))
            bpp_list.append(out_net["bpp"])

            out_criterion = criterion(epoch,out_net, current, lamada*weights[idx % 8])
            loss_i = out_criterion["loss"].mean()
            total_loss += loss_i

            idx += 1

        total_loss.backward()
        if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        

        if i % 500 == 0:
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_bpp = sum(bpp_list) / len(bpp_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            if dist.get_rank() == 0:
                logging.info(
                    f'[{i}/{len(train_dataloader.dataset)/gpu_per_batch/4}] | '
                    f'Multi-frame Loss: {total_loss.item():.3f} | '
                    f'PSNR: {avg_psnr:.3f} | '
                    f'SSIM: {avg_ssim.mean():.3f} | '
                    f'BPP: {avg_bpp.mean():.3f}'

                    )

def test_epoch(epoch,i_frame_net, test_dataloader, model, criterion, test_num):
    model.eval()
    device = next(model.parameters()).device
    i_frame_net = i_frame_net.to(device)
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            ref, input_images = Var(d[0]), Var(d[1])
            ref = ref.cuda(non_blocking=True)
            input_images = input_images.cuda(non_blocking=True)
            # input_images = list(input_images.split(3, dim=1))
            frame_num = len(input_images)
            
            lamada = 768
            qs_global = 71
            
            # 这里假设 `d` 是一个包含多个帧的序列，长度由 test_num 确定
            total_loss = 0.0
            total_bpp = 0.0
            total_psnr = 0.0
            if qs_global>63:
                ref = i_frame_net.compress_(ref, 63)
            else:
                ref = i_frame_net.compress_(ref, qs_global)
            model.module.clear_dpb()
            model.module.add_ref_frame(None, ref)
            for j in range(1, test_num + 1):
                
                # 对每个帧进行处理
                current = input_images[:, j, :, :, :]
                # fa_idx = index_map[j % 8]
                # curr_qp = model.module.shift_qp(qs_global, fa_idx)
                
                out_net = model(current, qs_global)  # 选择序列中的第 j 帧
                out_criterion = criterion(epoch,out_net, current, lamada)

                # 更新各个指标
                total_loss += out_criterion["loss"].mean()
                total_bpp += out_net["bpp"].mean()
                total_psnr += psnr(current, out_net["x_hat"])
               

            # 计算序列的平均损失和 PSNR
            avg_loss = total_loss / test_num
            avg_bpp = total_bpp / test_num
            avg_psnr = total_psnr / test_num

            # 更新指标的平均值
            loss_meter.update(avg_loss.item())
            psnr_meter.update(avg_psnr)
            bpp_meter.update(avg_bpp)
            mse_meter.update(out_criterion["mse_loss"].mean())
    if dist.get_rank() == 0:
        logging.info(
            f"Test epoch {epoch}: Average Loss: {loss_meter.avg:.3f} | "
            f"Average PSNR: {psnr_meter.avg:.3f} | "
            f"Average MSE: {mse_meter.avg:.8f} | "
            f"Average Bpp: {bpp_meter.avg:.4f}\n"
        )
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
    parser = argparse.ArgumentParser(description="Training script for DMCI model with multi-frame multi-stage training.")
    parser.add_argument("-m", "--model", default="DMC_slf_yuv420", choices=["DMC"], help="Model architecture")
    parser.add_argument("-td", "--test_dataset", type=str, default="/home/admin1/Data/data/testdata/HEVC_E/Johnny_1280x720_60", help="Training dataset path")
    parser.add_argument("-td_l", "--test_filelist", type=str, default="/home/admin1/Data/data/testdata/HEVC_E/image_paths.txt", help="Training dataset path")
    parser.add_argument("-e", "--epochs", default=180, type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Number of dataloader threads")
    parser.add_argument("-q", "--quality-level", type=int, default=2, help="Quality level")
    parser.add_argument("--lamada", type=float, default=1024, help="Rate-distortion parameter")
    parser.add_argument("--batch-size", type=int, default=4, help="Initial batch size (for single-frame training)")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Patch size (default: %(default)s)")
    parser.add_argument('--local-rank', default=-1, type=int,help='node rank for distributed training')
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument("--gpu-id", type=str, default=0, help="GPU id")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, help="Random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping max norm")
    parser.add_argument("--name", default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str, help="Result dir name")
    parser.add_argument("--model_path_i", type=str, default="checkpoints/cvpr2025_image.pth.tar", help="Path to checkpoint")
    parser.add_argument("--checkpoint", type=str, default="pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar", help="Path to checkpoint")
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

    if dist.get_rank() == 0:
        setup_logger(os.path.join(base_dir, time.strftime('%Y%m%d_%H%M%S') + '.log'))
        logging.info(f'======================= {args.name} =======================')
        for k, v in args.__dict__.items():
            logging.info(f'{k}: {v}')
        logging.info('=' * 40)

    # 构造测试数据（沿用 ImageFolder，不变）
    test_transforms = TetsDataSet(
        root=args.test_dataset,  # 数据集根目录
        filelist=args.test_filelist,  # 文件列表路径
        gop=32,  # 设置每组的帧数，用户自定义
        testfull=True  # 是否测试完整序列
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_transforms)

    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_transforms,  # 数据集对象
        batch_size=args.test_batch_size,  # 批处理大小
        num_workers=args.num_workers,  # 线程数
        pin_memory=(args.cuda and torch.cuda.is_available()),  # 如果使用 CUDA，启用内存固定
        sampler=test_sampler
    )

    # 使用代码2中定义的 DataSet 作为训练数据集
    train_dataset = DataSet()
    global_step = 0  # 全局训练步数（可根据实际迭代精确更新）
    # 初始化时默认处于单帧训练状态，batch size 使用 args.batch_size
    gpu_per_batch = args.batch_size
    test_num = 1  # 测试时每次只取一帧

    # 设置设备
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    # i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()
    # 实例化模型，并设置多GPU（如有）
    model = DMC()
    model = model.to(torch.device("cuda", args.local_rank))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True )

    # if args.cuda and torch.cuda.device_count() > 1:
    #     model = CustomDataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = RateDistortionLoss(lamada=args.lamada)

    last_epoch = 0
    if args.checkpoint:
        if dist.get_rank() == 0:
           logging.info("Loading checkpoint from %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    factors = [0.4, 0.1, 0.04, 0.01]  # 学习率衰减因子
    best_loss = float("inf")
    # 开始训练（按 epoch 循环）

    for epoch in range(last_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        logging.info(f"====== Current epoch {epoch} ======")
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Initialize num before conditions
        num = 2

        # 根据 global_step 动态更新数据集模式和 batch size
        if epoch < args.manyframe_epoch:
            if (epoch-50)>0:
               num = (epoch-50)/10 +3
            else :
               num = 3     
            num = int(num)
            if num>7:
                num = 7
            if num<=5:
                gpu_per_batch = args.batch_size
            else:
                gpu_per_batch = int(args.batch_size/2)
        else:
            num = 8 + (epoch-120)/2
            num  = int(num)
            if num > 32:
                num = 32
            gpu_per_batch = 1


        test_num = num-1  # 测试时每次只取一帧

        if test_num>30:
            test_num = 30


        train_dataset.set_frame_count(num)    # 更新数据集为1帧模式
        logging.info(f"Switched to {num} frame training dataset mode.")


        # 根据当前 gpu_per_batch 重构训练 DataLoader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=gpu_per_batch,
            num_workers=args.num_workers,
            sampler=train_sampler
        )
        
        # 训练一个 epoch（内部会根据 epoch 阶段控制帧数）
        
        train_one_epoch(epoch,model,i_frame_net, criterion, train_dataloader, optimizer,gpu_per_batch, args.clip_max_norm)
        loss = test_epoch(epoch,i_frame_net, test_dataloader, model, criterion,test_num)
        
        
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

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


