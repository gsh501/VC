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
import torch.optim as optim
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("DCVC_DISABLE_CUSTOMIZED_CUDA_INFERENCE", "1")

from src_uf.dataload_uf import UFDataSet, UFTestDataSet
from src_uf.models_uf.image_model import DCVCUFIntra

BEST_METRIC_FILENAMES = {
    "train_loss": "checkpoint_best_train_loss_uf_phase_1.pth.tar",
    "fixed_qp_loss": "checkpoint_best_qp63_loss_uf_phase_1.pth.tar",
    "multi_qp_loss": "checkpoint_best_multi_qp_loss_uf_phase_1.pth.tar",
    "average_bd_rate": "checkpoint_best_average_bd_rate_uf_phase_1.pth.tar",
    "low_qp_bd_rate": "checkpoint_best_low_qp_bd_rate_uf_phase_1.pth.tar",
}


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


def setup_distributed(args):
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)

    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.distributed:
        backend = "nccl" if args.cuda and torch.cuda.is_available() else "gloo"
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

    def forward(self, epoch, result, lambda_value):
        out = {
            "bpp_loss": result["bpp"],
            "mse_loss": result["mse"],
        }
        if 0 <= epoch < self.warmup_epochs:
            out["loss"] = out["mse_loss"] * 5000 + 0.01 * out["bpp_loss"]
        else:
            out["loss"] = lambda_value * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().mean().item()
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


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


def psnr_from_mse(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def unique_qps(values):
    return sorted({int(value) for value in values})


def init_best_metrics():
    return {metric: float("inf") for metric in BEST_METRIC_FILENAMES}


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


def checkpoint_metric_value(checkpoint, metric, default=float("inf")):
    if not isinstance(checkpoint, dict):
        return default

    best_metrics = checkpoint.get("best_metrics")
    if isinstance(best_metrics, dict) and metric in best_metrics:
        value = best_metrics[metric]
    elif metric == "fixed_qp_loss":
        value = checkpoint.get("best_fixed_qp_loss", checkpoint.get("best_loss", checkpoint.get("loss", default)))
    else:
        value = checkpoint.get("best_" + metric, checkpoint.get(metric, default))

    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_best_metrics_from_checkpoints(base_dir, best_metrics):
    for metric, filename in BEST_METRIC_FILENAMES.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        checkpoint = torch.load(path, map_location="cpu")
        best_metrics[metric] = min(
            best_metrics[metric],
            checkpoint_metric_value(checkpoint, metric, best_metrics[metric]),
        )


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

    best_metrics = init_best_metrics()
    for metric in best_metrics:
        best_metrics[metric] = checkpoint_metric_value(checkpoint, metric, best_metrics[metric])

    best_metrics["fixed_qp_loss"] = min(
        best_metrics["fixed_qp_loss"],
        checkpoint_loss_value(checkpoint, best_metrics["fixed_qp_loss"]),
    )
    load_best_metrics_from_checkpoints(os.path.dirname(checkpoint_path), best_metrics)
    bd_rate_anchor_curve = checkpoint.get("bd_rate_anchor_curve") if isinstance(checkpoint, dict) else None
    bd_rate_best_curves = None
    if isinstance(checkpoint, dict):
        bd_rate_best_curves = checkpoint.get("bd_rate_best_curves")
    if not isinstance(bd_rate_best_curves, dict):
        bd_rate_best_curves = init_bd_rate_best_curves()
        if bd_rate_anchor_curve is not None:
            bd_rate_best_curves["average_bd_rate"] = copy_rd_curve(bd_rate_anchor_curve)
            bd_rate_best_curves["low_qp_bd_rate"] = copy_rd_curve(bd_rate_anchor_curve)

    return checkpoint.get("epoch", -1) + 1, best_metrics, bd_rate_anchor_curve, bd_rate_best_curves


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
        lambda_value = qp_to_lambda(qp, q_num=args.q_num)

        out_net = model(ref_chunk, qp)
        out_criterion = criterion(epoch, out_net, lambda_value)
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
    return {
        "loss": avg_loss,
        "psnr": avg_psnr,
        "mse": avg_mse,
        "bpp": avg_bpp,
    }


def make_meter_group():
    return {
        "loss": AverageMeter(),
        "bpp": AverageMeter(),
        "mse": AverageMeter(),
    }


def reduce_val_meters(qp_meters, device):
    qp_metrics = {}
    for qp, meters in qp_meters.items():
        avg_loss = reduce_average_meter(meters["loss"], device)
        avg_mse = reduce_average_meter(meters["mse"], device)
        avg_bpp = reduce_average_meter(meters["bpp"], device)
        qp_metrics[qp] = {
            "loss": avg_loss,
            "psnr": psnr_from_mse(avg_mse),
            "mse": avg_mse,
            "bpp": avg_bpp,
        }
    return qp_metrics


def average_metric(qp_metrics, qps, key):
    values = [
        float(qp_metrics[qp][key])
        for qp in qps
        if qp in qp_metrics and math.isfinite(float(qp_metrics[qp][key]))
    ]
    if not values:
        return float("inf")
    return sum(values) / len(values)


def extract_rd_curve(qp_metrics, qps):
    return {
        int(qp): {
            "psnr": float(qp_metrics[qp]["psnr"]),
            "bpp": float(qp_metrics[qp]["bpp"]),
        }
        for qp in qps
        if (
            qp in qp_metrics
            and math.isfinite(float(qp_metrics[qp]["psnr"]))
            and math.isfinite(float(qp_metrics[qp]["bpp"]))
            and qp_metrics[qp]["bpp"] > 0
        )
    }


def interpolate_sorted(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for index in range(len(xs) - 1):
        left, right = xs[index], xs[index + 1]
        if left <= x <= right:
            ratio = (x - left) / max(right - left, 1e-12)
            return ys[index] + ratio * (ys[index + 1] - ys[index])
    return ys[-1]


def relative_rate_percent(current_curve, anchor_curve, qps):
    values = []
    for qp in qps:
        current = current_curve.get(qp)
        anchor = anchor_curve.get(qp)
        if (
            not current
            or not anchor
            or not math.isfinite(current["bpp"])
            or not math.isfinite(anchor["bpp"])
            or anchor["bpp"] <= 0
        ):
            continue
        values.append((current["bpp"] / anchor["bpp"] - 1.0) * 100.0)
    if not values:
        return float("inf")
    return sum(values) / len(values)


def bd_rate_percent(current_curve, anchor_curve, qps):
    current_points = [
        (current_curve[qp]["psnr"], math.log(max(current_curve[qp]["bpp"], 1e-12)))
        for qp in qps
        if qp in current_curve
    ]
    anchor_points = [
        (anchor_curve[qp]["psnr"], math.log(max(anchor_curve[qp]["bpp"], 1e-12)))
        for qp in qps
        if qp in anchor_curve
    ]
    if len(current_points) < 2 or len(anchor_points) < 2:
        return relative_rate_percent(current_curve, anchor_curve, qps)

    current_points = sorted(set(current_points))
    anchor_points = sorted(set(anchor_points))
    if len(current_points) < 2 or len(anchor_points) < 2:
        return relative_rate_percent(current_curve, anchor_curve, qps)

    current_psnr = [point[0] for point in current_points]
    current_log_rate = [point[1] for point in current_points]
    anchor_psnr = [point[0] for point in anchor_points]
    anchor_log_rate = [point[1] for point in anchor_points]
    left = max(min(current_psnr), min(anchor_psnr))
    right = min(max(current_psnr), max(anchor_psnr))
    if right <= left:
        return relative_rate_percent(current_curve, anchor_curve, qps)

    sample_count = 100
    delta_sum = 0.0
    for index in range(sample_count):
        ratio = index / max(sample_count - 1, 1)
        psnr_value = left + (right - left) * ratio
        current_value = interpolate_sorted(current_psnr, current_log_rate, psnr_value)
        anchor_value = interpolate_sorted(anchor_psnr, anchor_log_rate, psnr_value)
        delta_sum += current_value - anchor_value
    return (math.exp(delta_sum / sample_count) - 1.0) * 100.0


def copy_rd_curve(curve):
    if curve is None:
        return None
    return {int(qp): {"psnr": values["psnr"], "bpp": values["bpp"]} for qp, values in curve.items()}


def init_bd_rate_best_curves():
    return {
        "average_bd_rate": None,
        "low_qp_bd_rate": None,
    }


def update_bd_rate_best_curve(current_curve, best_curve, qps):
    if not current_curve:
        return best_curve, float("inf"), False
    if best_curve is None:
        return copy_rd_curve(current_curve), 0.0, True

    score = bd_rate_percent(current_curve, best_curve, qps)
    if score < 0:
        return copy_rd_curve(current_curve), score, True
    return best_curve, score, False


def add_bd_rate_metrics(val_metrics, anchor_curve, args):
    current_curve = extract_rd_curve(
        val_metrics["qps"],
        unique_qps(args.bd_rate_qps + args.low_qp_bd_rate_qps),
    )
    val_metrics["bd_rate_curve"] = current_curve
    if anchor_curve is None:
        anchor_curve = current_curve

    val_metrics["average_bd_rate"] = bd_rate_percent(
        current_curve,
        anchor_curve,
        args.bd_rate_qps,
    )
    val_metrics["low_qp_bd_rate"] = bd_rate_percent(
        current_curve,
        anchor_curve,
        args.low_qp_bd_rate_qps,
    )
    return anchor_curve


def val_epoch(epoch, model, criterion, val_dataloader, device, args):
    model.eval()
    qp_meters = {qp: make_meter_group() for qp in args.validation_qps}
    qp_lambdas = {qp: qp_to_lambda(qp, q_num=args.q_num) for qp in args.validation_qps}

    with torch.no_grad():
        for batch in val_dataloader:
            ref_chunk = batch[0].to(device, non_blocking=True)

            batch_size = ref_chunk.size(0)
            for qp in args.validation_qps:
                out_net = model(ref_chunk, qp)
                out_criterion = criterion(epoch, out_net, qp_lambdas[qp])
                loss = out_criterion["loss"].mean()
                qp_meters[qp]["loss"].update(loss, batch_size)
                qp_meters[qp]["bpp"].update(out_criterion["bpp_loss"], batch_size)
                qp_meters[qp]["mse"].update(out_criterion["mse_loss"], batch_size)

    qp_metrics = reduce_val_meters(qp_meters, device)
    fixed_qp_metrics = qp_metrics[args.test_qp]
    multi_qp_loss = average_metric(qp_metrics, args.multi_qp_loss_qps, "loss")

    if is_main_process():
        logging.info(
            "Val epoch %d: FixedQP(%d) Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f | MultiQPLoss: %.3f",
            epoch,
            args.test_qp,
            fixed_qp_metrics["loss"],
            fixed_qp_metrics["psnr"],
            fixed_qp_metrics["mse"],
            fixed_qp_metrics["bpp"],
            multi_qp_loss,
        )
    return {
        "loss": fixed_qp_metrics["loss"],
        "fixed_qp_loss": fixed_qp_metrics["loss"],
        "psnr": fixed_qp_metrics["psnr"],
        "mse": fixed_qp_metrics["mse"],
        "bpp": fixed_qp_metrics["bpp"],
        "multi_qp_loss": multi_qp_loss,
        "qps": qp_metrics,
    }


def save_checkpoint(state, best_filenames, base_dir, filename="checkpoint_uf_phase_1.pth.tar"):
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

    for best_filename in best_filenames:
        best_path = os.path.join(base_dir, best_filename)
        fd, tmp_best_path = tempfile.mkstemp(
            dir=base_dir,
            prefix=best_filename + ".tmp.",
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
    parser.add_argument("--val-qps", type=int, nargs="+", default=[0, 8, 16, 24, 32, 40, 48, 56, 63])
    parser.add_argument("--multi-qp-loss-qps", type=int, nargs="+", default=None)
    parser.add_argument("--bd-rate-qps", type=int, nargs="+", default=None)
    parser.add_argument("--low-qp-bd-rate-qps", type=int, nargs="+", default=[0, 8, 16, 24])
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
    if not args.train_filelist:
        raise ValueError("--train-filelist is required.")
    if args.val_dataset and not args.val_filelist:
        raise ValueError("--val-filelist is required when --val-dataset is provided.")
    if not (0 <= args.test_qp < args.q_num <= 64):
        raise ValueError("test_qp must be in [0, q_num), and q_num must be <= 64 for DCVCUFIntra.")
    args.val_qps = unique_qps(args.val_qps)
    args.multi_qp_loss_qps = unique_qps(args.multi_qp_loss_qps or args.val_qps)
    args.bd_rate_qps = unique_qps(args.bd_rate_qps or args.val_qps)
    args.low_qp_bd_rate_qps = unique_qps(args.low_qp_bd_rate_qps)
    args.validation_qps = unique_qps(
        [args.test_qp]
        + args.val_qps
        + args.multi_qp_loss_qps
        + args.bd_rate_qps
        + args.low_qp_bd_rate_qps
    )
    for name in ("val_qps", "multi_qp_loss_qps", "bd_rate_qps", "low_qp_bd_rate_qps", "validation_qps"):
        qps = getattr(args, name)
        if not qps:
            raise ValueError(f"{name} must not be empty.")
        if min(qps) < 0 or max(qps) >= args.q_num:
            raise ValueError(f"{name} must be in [0, q_num).")
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

    writer = None
    if is_main_process():
        tensorboard_dir = os.path.join(base_dir, "tensorboard", args.name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logging.info("TensorBoard logging to: %s", tensorboard_dir)

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
    best_metrics = init_best_metrics()
    bd_rate_anchor_curve = None
    bd_rate_best_curves = init_bd_rate_best_curves()
    if args.checkpoint:
        if is_main_process():
            logging.info("Loading checkpoint from %s", args.checkpoint)
        last_epoch, best_metrics, bd_rate_anchor_curve, bd_rate_best_curves = load_checkpoint(
            model,
            optimizer,
            args.checkpoint,
            device,
        )
        if is_main_process():
            logging.info(
                "Resume from epoch %d with best train_loss %.6f | fixed_qp_loss %.6f | multi_qp_loss %.6f | average_bd_rate %.6f | low_qp_bd_rate %.6f",
                last_epoch,
                best_metrics["train_loss"],
                best_metrics["fixed_qp_loss"],
                best_metrics["multi_qp_loss"],
                best_metrics["average_bd_rate"],
                best_metrics["low_qp_bd_rate"],
            )

    factors = [0.4, 0.1, 0.04, 0.01]
    for epoch in range(last_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        if is_main_process():
            logging.info("====== Current epoch %d ======", epoch)
            logging.info("Learning rate: %s", optimizer.param_groups[0]["lr"])

        train_metrics = train_one_epoch(
            epoch, model, criterion, train_dataloader, optimizer, device, args
        )

        if val_dataloader is not None:
            val_metrics = val_epoch(epoch, model, criterion, val_dataloader, device, args)
            bd_rate_anchor_curve = add_bd_rate_metrics(val_metrics, bd_rate_anchor_curve, args)
            if is_main_process():
                logging.info(
                    "Val epoch %d: AverageBDRate: %.6f | LowQPBDRate: %.6f",
                    epoch,
                    val_metrics["average_bd_rate"],
                    val_metrics["low_qp_bd_rate"],
                )
        else:
            val_metrics = {
                **train_metrics,
                "fixed_qp_loss": train_metrics["loss"],
                "multi_qp_loss": train_metrics["loss"],
                "average_bd_rate": float("inf"),
                "low_qp_bd_rate": float("inf"),
                "bd_rate_curve": {},
                "qps": {},
            }
            if is_main_process():
                logging.info("No val_filelist is set; using train loss for checkpoint selection.")

        current_metrics = {
            "train_loss": train_metrics["loss"],
            "fixed_qp_loss": val_metrics["fixed_qp_loss"],
            "multi_qp_loss": val_metrics["multi_qp_loss"],
        }
        best_filenames = []
        is_best_metrics = {metric: False for metric in BEST_METRIC_FILENAMES}
        for metric, value in current_metrics.items():
            is_best = math.isfinite(value) and value < best_metrics[metric]
            is_best_metrics[metric] = is_best
            if is_best:
                best_metrics[metric] = value
                best_filenames.append(BEST_METRIC_FILENAMES[metric])

        bd_rate_compare_scores = {}
        if val_dataloader is not None:
            bd_rate_metrics = (
                ("average_bd_rate", args.bd_rate_qps),
                ("low_qp_bd_rate", args.low_qp_bd_rate_qps),
            )
            current_bd_rate_curve = val_metrics.get("bd_rate_curve", {})
            for metric, qps in bd_rate_metrics:
                best_curve = bd_rate_best_curves.get(metric)
                updated_curve, score, is_best = update_bd_rate_best_curve(
                    current_bd_rate_curve,
                    best_curve,
                    qps,
                )
                bd_rate_compare_scores[metric] = score
                current_metrics[metric] = score
                val_metrics[f"{metric}_vs_best"] = score
                if is_best:
                    is_best_metrics[metric] = True
                    bd_rate_best_curves[metric] = updated_curve
                    best_metrics[metric] = score
                    best_filenames.append(BEST_METRIC_FILENAMES[metric])

        loss = current_metrics["fixed_qp_loss"]

        if is_main_process() and writer is not None:
            writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
            writer.add_scalar("Train/PSNR", train_metrics["psnr"], epoch)
            writer.add_scalar("Train/MSE", train_metrics["mse"], epoch)
            writer.add_scalar("Train/BPP", train_metrics["bpp"], epoch)

            if val_dataloader is not None:
                writer.add_scalar("Val/FixedQPLoss", val_metrics["fixed_qp_loss"], epoch)
                writer.add_scalar("Val/MultiQPLoss", val_metrics["multi_qp_loss"], epoch)
                writer.add_scalar("Val/AverageBDRate", val_metrics["average_bd_rate"], epoch)
                writer.add_scalar("Val/LowQPBDRate", val_metrics["low_qp_bd_rate"], epoch)
                for metric, tag in (
                    ("average_bd_rate", "Val/AverageBDRateVsBest"),
                    ("low_qp_bd_rate", "Val/LowQPBDRateVsBest"),
                ):
                    value = bd_rate_compare_scores.get(metric, float("inf"))
                    if math.isfinite(value):
                        writer.add_scalar(tag, value, epoch)
                writer.add_scalar("Val/PSNR", val_metrics["psnr"], epoch)
                writer.add_scalar("Val/MSE", val_metrics["mse"], epoch)
                writer.add_scalar("Val/BPP", val_metrics["bpp"], epoch)
                for qp, qp_metrics in val_metrics["qps"].items():
                    writer.add_scalar(f"ValQP/{qp}/Loss", qp_metrics["loss"], epoch)
                    writer.add_scalar(f"ValQP/{qp}/PSNR", qp_metrics["psnr"], epoch)
                    writer.add_scalar(f"ValQP/{qp}/MSE", qp_metrics["mse"], epoch)
                    writer.add_scalar(f"ValQP/{qp}/BPP", qp_metrics["bpp"], epoch)

            for metric, tag in (
                ("train_loss", "Best/TrainLoss"),
                ("fixed_qp_loss", "Best/FixedQPLoss"),
                ("multi_qp_loss", "Best/MultiQPLoss"),
                ("average_bd_rate", "Best/AverageBDRate"),
                ("low_qp_bd_rate", "Best/LowQPBDRate"),
            ):
                if math.isfinite(best_metrics[metric]):
                    writer.add_scalar(tag, best_metrics[metric], epoch)
            writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

        if args.save and is_main_process():
            checkpoint_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": loss,
                "train_loss": train_metrics["loss"],
                "best_loss": best_metrics["fixed_qp_loss"],
                "metrics": val_metrics,
                "train_metrics": train_metrics,
                "best_metrics": best_metrics,
                "bd_rate_anchor_curve": bd_rate_anchor_curve,
                "bd_rate_best_curves": bd_rate_best_curves,
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            save_checkpoint(
                checkpoint_state,
                best_filenames,
                base_dir,
            )
            logging.info(
                "Checkpoint epoch %d: best saved train_loss=%s fixed_qp_loss=%s multi_qp_loss=%s average_bd_rate=%s low_qp_bd_rate=%s | current: train_loss=%.6f fixed_qp_loss=%.6f multi_qp_loss=%.6f average_bd_rate=%.6f low_qp_bd_rate=%.6f | best: train_loss=%.6f fixed_qp_loss=%.6f multi_qp_loss=%.6f average_bd_rate=%.6f low_qp_bd_rate=%.6f",
                epoch,
                "yes" if is_best_metrics["train_loss"] else "no",
                "yes" if is_best_metrics["fixed_qp_loss"] else "no",
                "yes" if is_best_metrics["multi_qp_loss"] else "no",
                "yes" if is_best_metrics["average_bd_rate"] else "no",
                "yes" if is_best_metrics["low_qp_bd_rate"] else "no",
                current_metrics["train_loss"],
                current_metrics["fixed_qp_loss"],
                current_metrics["multi_qp_loss"],
                current_metrics.get("average_bd_rate", float("inf")),
                current_metrics.get("low_qp_bd_rate", float("inf")),
                best_metrics["train_loss"],
                best_metrics["fixed_qp_loss"],
                best_metrics["multi_qp_loss"],
                best_metrics["average_bd_rate"],
                best_metrics["low_qp_bd_rate"],
            )

        elif is_main_process():
            logging.info(
                "Checkpoint epoch %d: save disabled | best model saved: no | current fixed_qp_loss: %.6f | best fixed_qp_loss: %.6f",
                epoch,
                loss,
                best_metrics["fixed_qp_loss"],
            )

    if writer is not None:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main(sys.argv[1:])
