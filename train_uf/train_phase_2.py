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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src_uf.dataload_uf import UFDataSet, UFTestDataSet
from src_uf.models_uf.image_model import DCVCUFIntra
from src_uf.models_uf.video_model import DCVCUF


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
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
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


def choose_qp(epoch, step, args):
    if epoch < args.qp_warmup_epochs or step % 3 == 0:
        return args.q_num - 1
    return random.randint(0, args.q_num - 1)


class RateDistortionLoss(nn.Module):
    def __init__(self, warmup_epochs=20):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def forward(self, epoch, result, lamada):
        if epoch < self.warmup_epochs:
            return result["mse"] * 5000 + 0.01 * result["bpp"]
        return lamada * result["mse"] + result["bpp"]


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


def load_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint.get("net", checkpoint))
    consume_prefix_in_state_dict_if_present(state, "module.")
    return checkpoint, state


def load_intra_model(path, device):
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


def run_gop(intra_model, video_model, ref_chunk, input_chunks, qp, criterion, epoch, freeze_intra=False):
    if input_chunks.shape[1] != 3:
        raise RuntimeError(f"Expected three P chunks in a 32-frame GOP, got {input_chunks.shape[1]}")
    lamada = qp_to_lambda(qp)
    if freeze_intra:
        with torch.no_grad():
            intra_result = intra_model(ref_chunk, qp)
    else:
        intra_result = intra_model(ref_chunk, qp)
    core_model = unwrap(video_model)
    core_model.clear_dpb()
    core_model.set_curr_poc(0)
    core_model.add_ref_key_chunk(intra_result["x_hat"])

    losses = [criterion(epoch, intra_result, lamada).mean()]
    bpps = [intra_result["bpp"].mean()]
    mses = [intra_result["mse"].mean()]
    for chunk_index in range(input_chunks.shape[1]):
        result = video_model(input_chunks[:, chunk_index], qp)
        losses.append(criterion(epoch, result, lamada).mean())
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
            intra_model, model, ref_chunk, input_chunks, qp, criterion, epoch, args.freeze_intra)
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
    return reduce_meter(loss_meter, device), reduce_meter(bpp_meter, device), reduce_meter(mse_meter, device)

@torch.no_grad()
def val_epoch(epoch, model, intra_model, loader, criterion, device, args):
    model.eval()
    intra_model.eval()
    loss_meter, bpp_meter, mse_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for batch in loader:
        ref_chunk = batch[0].to(device, non_blocking=True)
        input_chunks = batch[1].to(device, non_blocking=True)
        loss, bpp, mse = run_gop(intra_model, model, ref_chunk, input_chunks,
                                  args.test_qp, criterion, epoch, args.freeze_intra)
        batch_size = ref_chunk.shape[0]
        loss_meter.update(loss, batch_size)
        bpp_meter.update(bpp, batch_size)
        mse_meter.update(mse, batch_size)
    return reduce_meter(loss_meter, device), reduce_meter(bpp_meter, device), reduce_meter(mse_meter, device)

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


def save_checkpoint(state, is_best, base_dir):
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
    if is_best:
        shutil.copyfile(path, os.path.join(base_dir, "checkpoint_best_loss_uf_phase_2.pth.tar"))


def save_model_weights(state, base_dir, filename, best_filename, is_best):
    path = os.path.join(base_dir, filename)
    descriptor, temporary = tempfile.mkstemp(dir=base_dir, prefix=filename + ".tmp.")
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    if is_best:
        shutil.copyfile(path, os.path.join(base_dir, best_filename))


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
    intra_model = load_intra_model(args.phase1_checkpoint, device)
    model = DCVCUF().to(device)
    start_epoch, best_loss = 0, float("inf")
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
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", best_loss)))
        if is_main_process():
            logging.info("Resume from epoch %d with best loss %.6f", start_epoch, best_loss)
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
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, factors)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        if is_main_process():
            logging.info("====== Current epoch %d ======", epoch)
            logging.info("Learning rate: %s", optimizer.param_groups[0]["lr"])
        train_loss, train_bpp, train_mse = train_epoch(
            epoch, model, intra_model, train_loader, optimizer, criterion, device, args)
        if is_main_process():
            logging.info(
                "Train epoch %d: Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f",
                epoch,
                train_loss,
                psnr_from_mse(train_mse),
                train_mse,
                train_bpp,
            )
        if val_loader is not None:
            loss, bpp, mse = val_epoch(epoch, model, intra_model, val_loader, criterion, device, args)
            if is_main_process():
                logging.info(
                    "Val epoch %d: Loss: %.3f | PSNR: %.3f | MSE: %.8f | BPP: %.4f",
                    epoch,
                    loss,
                    psnr_from_mse(mse),
                    mse,
                    bpp,
                )
        else:
            loss = train_loss
            bpp = train_bpp
            mse = train_mse
            if is_main_process():
                logging.info("No val_filelist is set; using train loss for checkpoint selection.")
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        if args.save and is_main_process():
            save_checkpoint({
                "epoch": epoch,
                "state_dict": unwrap(model).state_dict(),
                "intra_state_dict": unwrap(intra_model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
                "best_loss": best_loss,
                "args": vars(args),
            }, is_best, base_dir)
            save_model_weights(
                {"state_dict": unwrap(model).state_dict()},
                base_dir,
                "checkpoint_uf_phase_2_video.pth.tar",
                "checkpoint_best_loss_uf_phase_2_video.pth.tar",
                is_best,
            )
            save_model_weights(
                {"state_dict": unwrap(intra_model).state_dict()},
                base_dir,
                "checkpoint_uf_phase_2_intra.pth.tar",
                "checkpoint_best_loss_uf_phase_2_intra.pth.tar",
                is_best,
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
        global_step += len(train_loader)
        if is_main_process():
            logging.info("Global step updated to: %d", global_step)
    if is_dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
