import argparse
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("DCVC_DISABLE_CUSTOMIZED_CUDA_INFERENCE", "1")

from src_uf.dataload_uf import UFDataSet, UFTestDataSet
from src_uf.models_uf.image_model import DCVCUFIntra
from src_uf.utils.transforms import yuv_444_to_420


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_ready() else 0


def is_main_process():
    return get_rank() == 0

#分布式训练
def setup_distributed(args):
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)

    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if args.cuda and torch.cuda.is_available():
        if args.local_rank >= 0:
            torch.cuda.set_device(args.local_rank)
            return torch.device("cuda", args.local_rank)
        return torch.device("cuda")

    return torch.device("cpu")


def cleanup_distributed():
    if is_dist_ready():
        dist.destroy_process_group()


def adjust_learning_rate(optimizer, epoch, initial_lr, factors):
    lr = initial_lr
    if epoch >= 115:
        lr *= factors[3]
    elif epoch >= 110:
        lr *= factors[2]
    elif epoch >= 100:
        lr *= factors[1]
    elif epoch >= 60:
        lr *= factors[0]

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class RateDistortionLoss(nn.Module):
    def __init__(self, warmup_epochs=20):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def forward(self, epoch, result, target, lamada):
        out = {
            "bpp_loss": result["bpp"],
            "mse_loss": result["mse"],
        }
        if 0 <= epoch < self.warmup_epochs:
            out["loss"] = out["mse_loss"] * 5000 
        else:
            out["loss"] = lamada * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().mean().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

#保存训练模型路径
def init(args):
    base_dir = os.path.join(args.output_dir, args.model, str(args.quality_level))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info("Logging file is %s", log_path)


def calculate_psnr(x, x_hat, max_val=1.0):
    mse = F.mse_loss(x, x_hat, reduction="mean").clamp_min(1e-12)
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr

#一个chunk的psnr，未取均值
def psnr(x, x_hat, max_val=1.0):
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x_hat = x_hat.reshape(b * t, c, h, w)

    y_hat_420, uv_hat_420 = yuv_444_to_420(x_hat)
    y_420, uv_420 = yuv_444_to_420(x)
    u_420 = uv_420[:, 0:1, :, :]
    v_420 = uv_420[:, 1:2, :, :]
    u_hat_420 = uv_hat_420[:, 0:1, :, :]
    v_hat_420 = uv_hat_420[:, 1:2, :, :]

    psnr_y = calculate_psnr(y_420, y_hat_420, max_val)
    psnr_u = calculate_psnr(u_420, u_hat_420, max_val)
    psnr_v = calculate_psnr(v_420, v_hat_420, max_val)
    psnr = (6 * psnr_y + psnr_u + psnr_v) / 8.0
    return psnr


def psnr_from_mse(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def qp_to_lambda(qp, q_num=64, lam_min=1, lam_max=768):
    scale = qp / (q_num - 1)
    ln_lam_min = math.log(lam_min)
    ln_lam_max = math.log(lam_max)
    ln_lambda = ln_lam_min + scale * (ln_lam_max - ln_lam_min)
    return math.exp(ln_lambda)


def get_sync_random_value(epoch, i, q_num=64, warmup_epochs=48):
    if epoch < warmup_epochs:
        qp = q_num - 1
    elif i % 3 == 0:
        qp = q_num - 1
    else:
        qp = random.randint(0, q_num - 1)
    return qp


def sync_random_value(qp, device):
    if not is_dist_ready():
        return qp
    qp_tensor = torch.tensor(qp, device=device, dtype=torch.long)
    dist.broadcast(qp_tensor, src=0)
    return int(qp_tensor.item())

#所有GPU上得到的平均chunk的损失
def reduce_average_meter(meter, device):
    values = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total, count = values.tolist()
    return total / max(count, 1.0)


def checkpoint_loss_value(checkpoint, default=float("inf")):
    if not isinstance(checkpoint, dict):
        return default

    value = checkpoint.get("best_loss", checkpoint.get("loss", default))
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        target = model.module if hasattr(model, "module") else model
        consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
        target.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_loss = checkpoint_loss_value(checkpoint)
    best_checkpoint_path = os.path.join(
        os.path.dirname(checkpoint_path),
        "checkpoint_best_loss_uf_phase_1.pth.tar",
    )
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_loss = min(best_loss, checkpoint_loss_value(best_checkpoint, best_loss))

    return checkpoint.get("epoch", -1) + 1, best_loss


def train_one_epoch(epoch, model, criterion, train_dataloader, optimizer, device, args):
    model.train()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    for i, batch in enumerate(train_dataloader):
        ref_chunk = batch[0].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        qp = get_sync_random_value(epoch, i, args.q_num, args.qp_warmup_epochs)
        qp = sync_random_value(qp, device)
        lamada = qp_to_lambda(qp, q_num=args.q_num)

        out_net = model(ref_chunk, qp)
        out_criterion = criterion(epoch, out_net, ref_chunk, lamada)
        loss = out_criterion["loss"].mean()

        loss.backward()

        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        batch_size = ref_chunk.size(0)
        loss_meter.update(loss, batch_size)
        bpp_meter.update(out_criterion["bpp_loss"], batch_size)
        mse_meter.update(out_criterion["mse_loss"], batch_size)

        if is_main_process() and i % args.log_interval == 0:
            logging.info(
                "[%d/%d] | Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f | QP: %d",
                i,
                len(train_dataloader),
                loss_meter.avg,
                psnr_from_mse(mse_meter.avg),
                mse_meter.avg,
                bpp_meter.avg,
                qp,
            )

    avg_loss = reduce_average_meter(loss_meter, device)
    avg_mse = reduce_average_meter(mse_meter, device)
    avg_bpp = reduce_average_meter(bpp_meter, device)
    avg_psnr = psnr_from_mse(avg_mse)

    if is_main_process():
        logging.info(
            "Train epoch %d: Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f",
            epoch,
            avg_loss,
            avg_psnr,
            avg_mse,
            avg_bpp,
        )
    return avg_loss


def val_epoch(epoch, model, criterion, val_dataloader, device, args):
    model.eval()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    lamada = qp_to_lambda(args.test_qp, q_num=args.q_num)

    with torch.no_grad():
        for batch in val_dataloader:
            ref_chunk = batch[0].to(device, non_blocking=True)

            out_net = model(ref_chunk, args.test_qp)
            out_criterion = criterion(epoch, out_net, ref_chunk, lamada)
            loss = out_criterion["loss"].mean()

            batch_size = ref_chunk.size(0)
            loss_meter.update(loss, batch_size)
            bpp_meter.update(out_criterion["bpp_loss"], batch_size)
            mse_meter.update(out_criterion["mse_loss"], batch_size)

    avg_loss = reduce_average_meter(loss_meter, device)
    avg_mse = reduce_average_meter(mse_meter, device)
    avg_bpp = reduce_average_meter(bpp_meter, device)
    avg_psnr = psnr_from_mse(avg_mse)

    if is_main_process():
        logging.info(
            "Val epoch %d: Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f",
            epoch,
            avg_loss,
            avg_psnr,
            avg_mse,
            avg_bpp,
        )
    return avg_loss


def save_checkpoint(state, is_best, base_dir, filename="checkpoint_uf_phase_1.pth.tar"):
    path = os.path.join(base_dir, filename)

    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=filename + ".tmp.")
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if is_best:
        best_path = os.path.join(base_dir, "checkpoint_best_loss_uf_phase_1.pth.tar")
        fd, tmp_best_path = tempfile.mkstemp(
            dir=base_dir,
            prefix="checkpoint_best_loss_uf_phase_1.pth.tar.tmp.",
        )
        os.close(fd)
        try:
            shutil.copyfile(path, tmp_best_path)
            os.replace(tmp_best_path, best_path)
        except Exception:
            if os.path.exists(tmp_best_path):
                os.remove(tmp_best_path)
            raise


def build_train_loader(args):
    train_dataset = UFDataSet(
        rootdir=args.train_root,
        filefolderlist=args.train_filelist,
        im_height=args.patch_size[0],
        im_width=args.patch_size[1],
        chunk_size=args.chunk_size,
        gop=args.gop,
        check_exists=args.check_exists,
        max_samples=args.max_samples,
        pad_last=args.pad_last,
    )

    sampler = DistributedSampler(train_dataset) if args.distributed else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=(args.cuda and torch.cuda.is_available()),
        sampler=sampler,
        drop_last=args.drop_last,
        persistent_workers=args.num_workers > 0,
    )
    return dataloader, sampler


def build_val_loader(args):
    if not args.val_filelist:
        return None, None

    val_dataset = UFTestDataSet(
        root=args.val_dataset,
        filelist=args.val_filelist,
        gop=args.val_gop,
        chunk_size=args.chunk_size,
        testfull=True,
        pad_last=True,
    )
    sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.cuda and torch.cuda.is_available()),
        sampler=sampler,
        persistent_workers=args.num_workers > 0,
    )
    return dataloader, sampler


def parse_args(argv):
    parser = argparse.ArgumentParser(description="UF phase 1 training on 8-frame chunks.")
    parser.add_argument("-m", "--model", default="DCVCUFIntra", choices=["DCVCUFIntra"])
    parser.add_argument("--train-root", type=str, default=None)
    parser.add_argument("--train-filelist", type=str, default=None)
    parser.add_argument("-td", "--val-dataset", "--test-dataset", dest="val_dataset", type=str, default=None)
    parser.add_argument("-td_l", "--val-filelist", "--test-filelist", dest="val_filelist", type=str, default=None)
    parser.add_argument("-e", "--epochs", default=120, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=4)
    parser.add_argument("-q", "--quality-level", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", "--test-batch-size", dest="val_batch_size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument("--val-gop", "--test-gop", dest="val_gop", type=int, default=8)
    parser.add_argument("--q-num", type=int, default=64)
    parser.add_argument("--test-qp", type=int, default=63)
    parser.add_argument("--warmup-epochs", type=int, default=20)        #损失函数热身
    parser.add_argument("--qp-warmup-epochs", type=int, default=48)     #qp的热身
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=7)       #设置随机数种子
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", default=-1, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./pretrained_uf")
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), type=str)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--check-exists", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pad-last", type=str2bool, default=True)
    parser.add_argument("--drop-last", type=str2bool, default=False)
    args = parser.parse_args(argv)

    if args.chunk_size != 8 or args.gop != 8:
        raise ValueError("UF phase 1 trains the 8-frame intra chunk, so chunk_size and gop must both be 8.")
    if args.val_gop != 8:
        raise ValueError("UF phase 1 validation also uses val_gop=8.")
    if not (0 <= args.test_qp < args.q_num <= 64):
        raise ValueError("test_qp must be in [0, q_num), and q_num must be <= 64 for DCVCUFIntra.")
    return args


def main(argv):
    args = parse_args(argv)
    device = setup_distributed(args)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if is_main_process():
        setup_logger(os.path.join(base_dir, time.strftime("%Y%m%d_%H%M%S") + ".log"))
        logging.info("======================= %s =======================", args.name)
        for k, v in args.__dict__.items():
            logging.info("%s: %s", k, v)
        logging.info("=" * 40)

    train_dataloader, train_sampler = build_train_loader(args)
    val_dataloader, val_sampler = build_val_loader(args)

    model = DCVCUFIntra().to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = RateDistortionLoss(warmup_epochs=args.warmup_epochs)

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:
        if is_main_process():
            logging.info("Loading checkpoint from %s", args.checkpoint)
        last_epoch, best_loss = load_checkpoint(model, optimizer, args.checkpoint, device)
        if is_main_process():
            logging.info("Resume from epoch %d with best loss %.6f", last_epoch, best_loss)

    factors = [0.4, 0.1, 0.04, 0.01]
    global_step = last_epoch * len(train_dataloader)

    for epoch in range(last_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        if is_main_process():
            logging.info("====== Current epoch %d ======", epoch)
            logging.info("Learning rate: %s", optimizer.param_groups[0]["lr"])

        train_loss = train_one_epoch(
            epoch, model, criterion, train_dataloader, optimizer, device, args
        )

        if val_dataloader is not None:
            loss = val_epoch(epoch, model, criterion, val_dataloader, device, args)
        else:
            loss = train_loss
            if is_main_process():
                logging.info("No val_filelist is set; using train loss for checkpoint selection.")

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and is_main_process():
            checkpoint_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": loss,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            save_checkpoint(
                checkpoint_state,
                is_best,
                base_dir,
            )
            logging.info(
                "Checkpoint epoch %d: best model saved: %s | current loss: %.6f | best loss: %.6f",
                epoch,
                "yes" if is_best else "no",
                loss,
                best_loss,
            )

        elif is_main_process():
            logging.info(
                "Checkpoint epoch %d: save disabled | best model saved: no | current loss: %.6f | best loss: %.6f",
                epoch,
                loss,
                best_loss,
            )

        global_step += len(train_dataloader)
        if is_main_process():
            logging.info("Global step updated to: %d", global_step)

    cleanup_distributed()


if __name__ == "__main__":
    main(sys.argv[1:])
