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

from src_uf.dataload_uf import UFDataSet, UFTestDataSet
from src_uf.models_uf.image_model import DCVCUFIntra
from src_uf.models_uf.video_model import DCVCUF

BEST_METRIC_FILENAMES = {
    "train_loss": "checkpoint_best_train_loss_uf_phase_2.pth.tar",
    "fixed_qp_loss": "checkpoint_best_qp63_loss_uf_phase_2.pth.tar",
    "multi_qp_loss": "checkpoint_best_multi_qp_loss_uf_phase_2.pth.tar",
    "average_bd_rate": "checkpoint_best_average_bd_rate_uf_phase_2.pth.tar",
    "low_qp_bd_rate": "checkpoint_best_low_qp_bd_rate_uf_phase_2.pth.tar",
}
BEST_SPLIT_METRIC_FILENAMES = {
    metric: {
        "video": filename.replace(".pth.tar", "_video.pth.tar"),
        "intra": filename.replace(".pth.tar", "_intra.pth.tar"),
    }
    for metric, filename in BEST_METRIC_FILENAMES.items()
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("yes", "y", "true", "t", "1")


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_dist_ready() or dist.get_rank() == 0


def setup_distributed(args):
    if os.environ.get("LOCAL_RANK") is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
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


def adjust_learning_rate(optimizer, epoch, initial_lr, factors):
    lr = initial_lr
    if epoch >= 175:
        lr *= factors[3]
    elif epoch >= 170:
        lr *= factors[2]
    elif epoch >= 160:
        lr *= factors[1]
    elif epoch >= 100:
        lr *= factors[0]

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def unwrap(model):
    return model.module if hasattr(model, "module") else model

def trainable_parameters(*models):
    for model in models:
        for param in unwrap(model).parameters():
            if param.requires_grad:
                yield param


def sync_qp(qp, device):
    if not is_dist_ready():
        return qp
    value = torch.tensor(qp, device=device, dtype=torch.long)
    dist.broadcast(value, src=0)
    return int(value.item())


def qp_to_lambda(qp, q_num=64, lam_min=1, lam_max=768):
    scale = qp / (q_num - 1)
    return math.exp(math.log(lam_min) + scale * (math.log(lam_max) - math.log(lam_min)))


def psnr_from_mse(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def unique_qps(values):
    return sorted({int(value) for value in values})


def init_best_metrics():
    return {metric: float("inf") for metric in BEST_METRIC_FILENAMES}


def checkpoint_metric_value(checkpoint, metric, default=float("inf")):
    if not isinstance(checkpoint, dict):
        return default

    best_metrics = checkpoint.get("best_metrics")
    if isinstance(best_metrics, dict) and metric in best_metrics:
        value = best_metrics[metric]
    elif metric == "fixed_qp_loss":
        value = checkpoint.get("best_loss", checkpoint.get("loss", default))
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


def choose_qp(epoch, step, args):
    if epoch < args.qp_warmup_epochs or step % 3 == 0:
        return args.q_num - 1
    return random.randint(0, args.q_num - 1)


class RateDistortionLoss(nn.Module):
    def __init__(self, warmup_epochs=20):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def forward(self, epoch, result, lambda_value):
        if epoch < self.warmup_epochs:
            return result["mse"] * 5000 + 0.01 * result["bpp"]
        return lambda_value * result["mse"] + result["bpp"]


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        if torch.is_tensor(value):
            value = value.detach().mean().item()
        self.sum += value * count
        self.count += count

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def reduce_meter(meter, device):
    values = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(values)
    return values[0].item() / max(values[1].item(), 1.0)


def reduce_qp_meters(qp_meters, device):
    qp_metrics = {}
    for qp, meters in qp_meters.items():
        loss = reduce_meter(meters["loss"], device)
        bpp = reduce_meter(meters["bpp"], device)
        mse = reduce_meter(meters["mse"], device)
        qp_metrics[qp] = {
            "loss": loss,
            "psnr": psnr_from_mse(mse),
            "mse": mse,
            "bpp": bpp,
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
    return {
        int(qp): {
            "psnr": float(values["psnr"]),
            "bpp": float(values["bpp"]),
        }
        for qp, values in curve.items()
    }


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


def load_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint.get("net", checkpoint))
    consume_prefix_in_state_dict_if_present(state, "module.")
    return checkpoint, state


def load_intra_model(path, device):
    if path is None:
        raise FileNotFoundError("Phase-1 checkpoint path is required.")
    if not os.path.isfile(path):
        legacy_name = "checkpoint_best_loss_uf_phase_1.pth.tar"
        current_name = "checkpoint_best_qp63_loss_uf_phase_1.pth.tar"
        if str(path).endswith(legacy_name):
            fallback_path = str(path)[: -len(legacy_name)] + current_name
            if os.path.isfile(fallback_path):
                path = fallback_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Phase-1 checkpoint not found: {path}")
    _, state = load_state(path, device)
    model = DCVCUFIntra().to(device)
    model.load_state_dict(state)
    model.train()
    model.requires_grad_(True)
    return model


def configure_intra_model(intra_model, freeze_intra):
    intra_model.requires_grad_(not freeze_intra)
    if freeze_intra:
        intra_model.eval()
    else:
        intra_model.train()


def run_gop(
    intra_model,
    video_model,
    ref_chunk,
    input_chunks,
    qp,
    criterion,
    epoch,
    q_num=64,
    freeze_intra=False,
):
    if input_chunks.shape[1] != 3:
        raise RuntimeError(f"Expected three P chunks in a 32-frame GOP, got {input_chunks.shape[1]}")
    lambda_value = qp_to_lambda(qp, q_num=q_num)
    if freeze_intra:
        with torch.no_grad():
            intra_result = intra_model(ref_chunk, qp)
    else:
        intra_result = intra_model(ref_chunk, qp)
    core_model = unwrap(video_model)
    core_model.clear_dpb()
    core_model.set_curr_poc(0)
    core_model.add_ref_key_chunk(intra_result["x_hat"])

    losses = [criterion(epoch, intra_result, lambda_value).mean()]
    bpps = [intra_result["bpp"].mean()]
    mses = [intra_result["mse"].mean()]
    for chunk_index in range(input_chunks.shape[1]):
        result = video_model(input_chunks[:, chunk_index], qp)
        losses.append(criterion(epoch, result, lambda_value).mean())
        bpps.append(result["bpp"].mean())
        mses.append(result["mse"].mean())
    return torch.stack(losses).mean(), torch.stack(bpps).mean(), torch.stack(mses).mean()

def train_epoch(epoch, model, intra_model, loader, optimizer, criterion, device, args):
    model.train()
    if args.freeze_intra:
        intra_model.eval()
    else:
        intra_model.train()
    loss_meter, bpp_meter, mse_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for step, batch in enumerate(loader):
        ref_chunk = batch[0].to(device, non_blocking=True)
        input_chunks = batch[1].to(device, non_blocking=True)
        qp = sync_qp(choose_qp(epoch, step, args), device)
        optimizer.zero_grad(set_to_none=True)
        loss, bpp, mse = run_gop(
            intra_model,
            model,
            ref_chunk,
            input_chunks,
            qp,
            criterion,
            epoch,
            q_num=args.q_num,
            freeze_intra=args.freeze_intra,
        )
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model, intra_model), args.clip_max_norm)
        optimizer.step()
        batch_size = ref_chunk.shape[0]
        loss_meter.update(loss, batch_size)
        bpp_meter.update(bpp, batch_size)
        mse_meter.update(mse, batch_size)
        if is_main_process() and step % args.log_interval == 0:
            logging.info(
                "[%d/%d] | Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f | QP: %d",
                step,
                len(loader),
                loss_meter.avg,
                psnr_from_mse(mse_meter.avg),
                mse_meter.avg,
                bpp_meter.avg,
                qp,
            )
    loss = reduce_meter(loss_meter, device)
    bpp = reduce_meter(bpp_meter, device)
    mse = reduce_meter(mse_meter, device)
    return {
        "loss": loss,
        "psnr": psnr_from_mse(mse),
        "mse": mse,
        "bpp": bpp,
    }

@torch.no_grad()
def val_epoch(epoch, model, intra_model, loader, criterion, device, args):
    model.eval()
    intra_model.eval()
    qp_meters = {
        qp: {
            "loss": AverageMeter(),
            "bpp": AverageMeter(),
            "mse": AverageMeter(),
        }
        for qp in args.validation_qps
    }
    for batch in loader:
        ref_chunk = batch[0].to(device, non_blocking=True)
        input_chunks = batch[1].to(device, non_blocking=True)
        batch_size = ref_chunk.shape[0]
        for qp in args.validation_qps:
            loss, bpp, mse = run_gop(
                intra_model,
                model,
                ref_chunk,
                input_chunks,
                qp,
                criterion,
                epoch,
                q_num=args.q_num,
                freeze_intra=args.freeze_intra,
            )
            qp_meters[qp]["loss"].update(loss, batch_size)
            qp_meters[qp]["bpp"].update(bpp, batch_size)
            qp_meters[qp]["mse"].update(mse, batch_size)

    qp_metrics = reduce_qp_meters(qp_meters, device)
    fixed_qp_metrics = qp_metrics[args.test_qp]
    return {
        "loss": fixed_qp_metrics["loss"],
        "fixed_qp_loss": fixed_qp_metrics["loss"],
        "psnr": fixed_qp_metrics["psnr"],
        "mse": fixed_qp_metrics["mse"],
        "bpp": fixed_qp_metrics["bpp"],
        "multi_qp_loss": average_metric(qp_metrics, args.multi_qp_loss_qps, "loss"),
        "qps": qp_metrics,
    }


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

def build_loaders(args):
    train_set = UFDataSet(rootdir=args.train_root, filefolderlist=args.train_filelist,
                          im_height=args.patch_size[0], im_width=args.patch_size[1],
                          chunk_size=8, gop=32, check_exists=args.check_exists,
                          max_samples=args.max_samples, pad_last=False)
    train_set.gops = [gop for gop in train_set.gops if len(gop) == 32]
    if not train_set.gops:
        raise RuntimeError("No complete 32-frame GOP was found in the training file list.")
    train_sampler = DistributedSampler(train_set) if args.distributed else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=args.cuda and torch.cuda.is_available(),
                              drop_last=args.drop_last, persistent_workers=args.num_workers > 0)
    if not args.val_filelist:
        return train_loader, train_sampler, None, None
    val_set = UFTestDataSet(root=args.val_dataset, filelist=args.val_filelist,
                            gop=32, chunk_size=8, testfull=True, pad_last=False)
    complete_indices = [index for index, gop in enumerate(val_set.gops) if len(gop) == 32]
    val_set.gops = [val_set.gops[index] for index in complete_indices]
    val_set.image_names = [val_set.image_names[index] for index in complete_indices]
    if not val_set.gops:
        raise RuntimeError("No complete 32-frame GOP was found in the validation file list.")
    val_sampler = DistributedSampler(val_set, shuffle=False) if args.distributed else None
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, shuffle=False,
                            pin_memory=args.cuda and torch.cuda.is_available(),
                            persistent_workers=args.num_workers > 0)
    return train_loader, train_sampler, val_loader, val_sampler


def copy_checkpoint(source_path, target_path):
    target_dir = os.path.dirname(target_path)
    filename = os.path.basename(target_path)
    descriptor, temporary = tempfile.mkstemp(
        dir=target_dir,
        prefix=filename + ".tmp.",
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, target_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_checkpoint(state, best_filenames, base_dir):
    filename = "checkpoint_uf_phase_2.pth.tar"
    path = os.path.join(base_dir, filename)
    descriptor, temporary = tempfile.mkstemp(dir=base_dir, prefix=filename + ".tmp.")
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    for best_filename in best_filenames:
        copy_checkpoint(path, os.path.join(base_dir, best_filename))


def save_model_weights(state, base_dir, filename, best_filenames):
    path = os.path.join(base_dir, filename)
    descriptor, temporary = tempfile.mkstemp(dir=base_dir, prefix=filename + ".tmp.")
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    for best_filename in best_filenames:
        copy_checkpoint(path, os.path.join(base_dir, best_filename))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="UF phase 2 training on 32-frame (8 I + 3x8 P) chunks.")
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--train-filelist", required=True)
    parser.add_argument("--val-dataset", "--test-dataset", "-td", dest="val_dataset", default=None)
    parser.add_argument("--val-filelist", "--test-filelist", "-td_l", dest="val_filelist", default=None)
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="./pretrained_uf")
    parser.add_argument("--quality-level", "-q", type=int, default=1)
    parser.add_argument("--epochs", "-e", type=int, default=180)
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", "-n", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", "--test-batch-size", dest="val_batch_size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--q-num", type=int, default=64)
    parser.add_argument("--val-qp", "--test-qp", dest="test_qp", type=int, default=63)
    parser.add_argument(
        "--val-qps",
        type=int,
        nargs="+",
        default=[0, 8, 16, 24, 32, 40, 48, 56, 63],
    )
    parser.add_argument("--multi-qp-loss-qps", type=int, nargs="+", default=None)
    parser.add_argument("--bd-rate-qps", type=int, nargs="+", default=None)
    parser.add_argument("--low-qp-bd-rate-qps", type=int, nargs="+", default=[0, 8, 16, 24])
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--qp-warmup-epochs", type=int, default=48)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--freeze-intra", "--freeze-dcvcufintra", dest="freeze_intra",
                        type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--check-exists", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--drop-last", type=str2bool, default=False)
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    args = parser.parse_args(argv)
    if not (0 <= args.test_qp < args.q_num <= 64):
        parser.error("val_qp/test_qp must be in [0, q_num), with q_num <= 64")
    if not args.phase1_checkpoint:
        parser.error("--phase1-checkpoint is required.")
    if args.val_dataset and not args.val_filelist:
        parser.error("--val-filelist is required when --val-dataset is provided.")

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
    for name in (
        "val_qps",
        "multi_qp_loss_qps",
        "bd_rate_qps",
        "low_qp_bd_rate_qps",
        "validation_qps",
    ):
        qps = getattr(args, name)
        if not qps:
            parser.error(f"{name} must not be empty.")
        if min(qps) < 0 or max(qps) >= args.q_num:
            parser.error(f"{name} must be in [0, q_num).")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = setup_distributed(args)
    base_dir = os.path.join(args.output_dir, "DCVCUF", str(args.quality_level))
    os.makedirs(base_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if is_main_process():
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[logging.FileHandler(os.path.join(base_dir, time.strftime("%Y%m%d_%H%M%S") + ".log"), encoding="utf-8"),
                                      logging.StreamHandler(sys.stdout)], force=True)
        logging.info("======================= %s =======================", args.name)
        for key, value in vars(args).items():
            logging.info("%s: %s", key, value)

    train_loader, train_sampler, val_loader, val_sampler = build_loaders(args)
    writer = None
    if is_main_process():
        tensorboard_dir = os.path.join(base_dir, "tensorboard", args.name)
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logging.info("TensorBoard logging to: %s", tensorboard_dir)

    intra_model = load_intra_model(args.phase1_checkpoint, device)
    model = DCVCUF().to(device)
    start_epoch = 0
    best_metrics = init_best_metrics()
    bd_rate_anchor_curve = None
    bd_rate_best_curves = init_bd_rate_best_curves()
    checkpoint = None
    if args.checkpoint:
        if is_main_process():
            logging.info("Loading checkpoint from %s", args.checkpoint)
        checkpoint, state = load_state(args.checkpoint, device)
        model.load_state_dict(state)
        intra_state = checkpoint.get("intra_state_dict")
        if intra_state is not None:
            consume_prefix_in_state_dict_if_present(intra_state, "module.")
            intra_model.load_state_dict(intra_state)
        start_epoch = checkpoint.get("epoch", -1) + 1
        for metric in best_metrics:
            best_metrics[metric] = checkpoint_metric_value(
                checkpoint,
                metric,
                best_metrics[metric],
            )
        load_best_metrics_from_checkpoints(os.path.dirname(args.checkpoint), best_metrics)
        bd_rate_anchor_curve = checkpoint.get("bd_rate_anchor_curve")
        saved_best_curves = checkpoint.get("bd_rate_best_curves")
        if isinstance(saved_best_curves, dict):
            bd_rate_best_curves = saved_best_curves
        elif bd_rate_anchor_curve is not None:
            bd_rate_best_curves["average_bd_rate"] = copy_rd_curve(bd_rate_anchor_curve)
            bd_rate_best_curves["low_qp_bd_rate"] = copy_rd_curve(bd_rate_anchor_curve)
        if is_main_process():
            logging.info(
                "Resume from epoch %d with best train_loss %.6f | fixed_qp_loss %.6f | multi_qp_loss %.6f | average_bd_rate %.6f | low_qp_bd_rate %.6f",
                start_epoch,
                best_metrics["train_loss"],
                best_metrics["fixed_qp_loss"],
                best_metrics["multi_qp_loss"],
                best_metrics["average_bd_rate"],
                best_metrics["low_qp_bd_rate"],
            )
    configure_intra_model(intra_model, args.freeze_intra)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True)
        if not args.freeze_intra:
            intra_model = torch.nn.parallel.DistributedDataParallel(
                intra_model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True)
    optimizer = optim.AdamW(trainable_parameters(model, intra_model), lr=args.learning_rate)
    if checkpoint is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, RuntimeError) as exc:
            if is_main_process():
                logging.warning("Skipping optimizer state restore: %s", exc)
    trainable_count = sum(p.numel() for p in trainable_parameters(model, intra_model))
    total_count = sum(p.numel() for p in list(unwrap(model).parameters()) + list(unwrap(intra_model).parameters()))
    if is_main_process():
        logging.info("Trainable params: %d / %d", trainable_count, total_count)
    criterion = RateDistortionLoss(args.warmup_epochs)
    factors = [0.4, 0.1, 0.04, 0.01]

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        if is_main_process():
            logging.info("====== Current epoch %d ======", epoch)
            logging.info("Learning rate: %s", optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(
            epoch,
            model,
            intra_model,
            train_loader,
            optimizer,
            criterion,
            device,
            args,
        )
        if is_main_process():
            logging.info(
                "Train epoch %d: Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["psnr"],
                train_metrics["mse"],
                train_metrics["bpp"],
            )
        if val_loader is not None:
            val_metrics = val_epoch(
                epoch,
                model,
                intra_model,
                val_loader,
                criterion,
                device,
                args,
            )
            bd_rate_anchor_curve = add_bd_rate_metrics(
                val_metrics,
                bd_rate_anchor_curve,
                args,
            )
            if is_main_process():
                logging.info(
                    "Val epoch %d: FixedQP(%d) Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f | MultiQPLoss: %.3f | AverageBDRate: %.6f | LowQPBDRate: %.6f",
                    epoch,
                    args.test_qp,
                    val_metrics["fixed_qp_loss"],
                    val_metrics["psnr"],
                    val_metrics["mse"],
                    val_metrics["bpp"],
                    val_metrics["multi_qp_loss"],
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
        best_metric_names = []
        is_best_metrics = {metric: False for metric in BEST_METRIC_FILENAMES}
        for metric, value in current_metrics.items():
            is_best = math.isfinite(value) and value < best_metrics[metric]
            is_best_metrics[metric] = is_best
            if is_best:
                best_metrics[metric] = value
                best_metric_names.append(metric)

        bd_rate_compare_scores = {}
        if val_loader is not None:
            for metric, qps in (
                ("average_bd_rate", args.bd_rate_qps),
                ("low_qp_bd_rate", args.low_qp_bd_rate_qps),
            ):
                best_curve = bd_rate_best_curves.get(metric)
                updated_curve, score, is_best = update_bd_rate_best_curve(
                    val_metrics["bd_rate_curve"],
                    best_curve,
                    qps,
                )
                bd_rate_compare_scores[metric] = score
                current_metrics[metric] = score
                val_metrics[f"{metric}_vs_best"] = score
                if is_best:
                    is_best_metrics[metric] = True
                    best_metrics[metric] = score
                    bd_rate_best_curves[metric] = updated_curve
                    best_metric_names.append(metric)

        best_filenames = [
            BEST_METRIC_FILENAMES[metric] for metric in best_metric_names
        ]
        loss = current_metrics["fixed_qp_loss"]

        if is_main_process() and writer is not None:
            writer.add_scalar("Train/Loss", train_metrics["loss"], epoch)
            writer.add_scalar("Train/PSNR", train_metrics["psnr"], epoch)
            writer.add_scalar("Train/MSE", train_metrics["mse"], epoch)
            writer.add_scalar("Train/BPP", train_metrics["bpp"], epoch)
            if val_loader is not None:
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
                "state_dict": unwrap(model).state_dict(),
                "intra_state_dict": unwrap(intra_model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
                "train_loss": train_metrics["loss"],
                "best_loss": best_metrics["fixed_qp_loss"],
                "metrics": val_metrics,
                "train_metrics": train_metrics,
                "best_metrics": best_metrics,
                "bd_rate_anchor_curve": bd_rate_anchor_curve,
                "bd_rate_best_curves": bd_rate_best_curves,
                "args": vars(args),
            }
            save_checkpoint(checkpoint_state, best_filenames, base_dir)
            video_best_filenames = [
                BEST_SPLIT_METRIC_FILENAMES[metric]["video"]
                for metric in best_metric_names
            ]
            intra_best_filenames = [
                BEST_SPLIT_METRIC_FILENAMES[metric]["intra"]
                for metric in best_metric_names
            ]
            save_model_weights(
                {"state_dict": unwrap(model).state_dict()},
                base_dir,
                "checkpoint_uf_phase_2_video.pth.tar",
                video_best_filenames,
            )
            save_model_weights(
                {"state_dict": unwrap(intra_model).state_dict()},
                base_dir,
                "checkpoint_uf_phase_2_intra.pth.tar",
                intra_best_filenames,
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
                "Checkpoint epoch %d: save disabled | current fixed_qp_loss: %.6f | best fixed_qp_loss: %.6f",
                epoch,
                loss,
                best_metrics["fixed_qp_loss"],
            )
    if writer is not None:
        writer.close()
    if is_dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
