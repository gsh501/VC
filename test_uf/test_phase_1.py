import argparse
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_uf.dataload_uf import UFTestDataSet
from src_uf.models_uf.image_model import DCVCUFIntra
from src_uf.utils.transforms import ycbcr2rgb, yuv_444_to_420


DEFAULT_TEST_DATASET = Path("/home/admin1/Data/data/testdata")
DEFAULT_TEST_CLASSES = ["HEVC_B", "HEVC_C", "HEVC_D", "HEVC_E"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "y", "true", "t", "1")


class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().mean().item()
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def psnr_from_mse(mse):
    return -10.0 * math.log10(max(float(mse), 1e-12))


def calculate_psnr(x, x_hat, max_val=1.0):
    mse = F.mse_loss(x, x_hat, reduction="mean").clamp_min(1e-12)
    return 10 * torch.log10(max_val ** 2 / mse)


def yuv420_weighted_psnr(x, x_hat, max_val=1.0):
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
    return (6 * psnr_y + psnr_u + psnr_v) / 8.0


def pad_chunk_to_multiple(x, multiple):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w

    b, t, c, h, w = x.shape
    flat = x.reshape(b * t, c, h, w)
    flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    return flat.reshape(b, t, c, h + pad_h, w + pad_w), h, w


def crop_chunk(x, height, width):
    return x[..., :height, :width].contiguous()


def chunk_mse(x, x_hat):
    mse = F.mse_loss(x_hat, x, reduction="none")
    return mse.flatten(1).mean(dim=1)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def default_log_path():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"test_phase_1_{timestamp}.log"


def setup_stdout_log(log_file):
    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_handle)
    return log_path, log_handle, original_stdout


def default_output_dir(log_path):
    return Path(__file__).resolve().parent / "codec_outputs" / Path(log_path).stem


def sanitize_path_part(text):
    text = str(text).replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "checkpoint"


def maybe_synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_label(checkpoint_path, checkpoint_dir):
    checkpoint_path = Path(checkpoint_path).resolve()
    for base in (Path(checkpoint_dir).resolve(), REPO_ROOT):
        try:
            return checkpoint_path.relative_to(base).as_posix()
        except ValueError:
            pass
    return checkpoint_path.as_posix()


def disable_custom_cuda_inference():
    import src_uf.layers.cuda_inference as cuda_inference
    import src_uf.layers.layers as layer_ops
    import src_uf.models_uf.image_model as image_model

    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    layer_ops.CUSTOMIZED_CUDA_INFERENCE = False
    image_model.CUSTOMIZED_CUDA_INFERENCE = False


def set_custom_cuda_kernel_enabled(
    depthconv_enabled,
    subpel_enabled,
    round_int8_enabled,
    clamp_reciprocal_enabled,
    build_index_enabled,
):
    import src_uf.layers.cuda_inference as cuda_inference
    import src_uf.layers.layers as layer_ops

    layer_ops.DISABLE_DEPTHCONV_PROXY = not depthconv_enabled
    layer_ops.DISABLE_SUBPEL_PROXY = not subpel_enabled
    cuda_inference.DISABLE_ROUND_AND_TO_INT8_CUDA = not round_int8_enabled
    cuda_inference.DISABLE_CLAMP_RECIPROCAL_CUDA = not clamp_reciprocal_enabled
    cuda_inference.DISABLE_BUILD_INDEX_CUDA = not build_index_enabled


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("net", checkpoint))

    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def frame_sort_key(path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    frame_index = int(digits) if digits else 0
    return frame_index, path.name


def folder_has_frames(folder):
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in folder.iterdir()
    )


def collect_sequence_dirs(test_dataset, test_classes):
    root = Path(test_dataset).resolve()
    class_dirs = [root / class_name for class_name in test_classes]
    if test_classes and any(class_dir.exists() for class_dir in class_dirs):
        search_dirs = [root / class_name for class_name in test_classes]
    else:
        search_dirs = [root]

    sequence_dirs = []
    missing_dirs = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            missing_dirs.append(search_dir)
            continue

        if folder_has_frames(search_dir):
            sequence_dirs.append(search_dir)
            continue

        for child in sorted(search_dir.iterdir()):
            if child.is_dir() and folder_has_frames(child):
                sequence_dirs.append(child)

    if missing_dirs:
        missing_text = "\n".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"These test class folders were not found:\n{missing_text}")
    if not sequence_dirs:
        raise RuntimeError(f"No sequence folders with image frames found under: {root}")

    return sorted(sequence_dirs)


def collect_test_frames(test_dataset, test_classes, expected_frames, allow_incomplete):
    sequence_dirs = collect_sequence_dirs(test_dataset, test_classes)
    frames = []
    bad_sequences = []

    for sequence_dir in sequence_dirs:
        sequence_frames = sorted(
            [
                path.resolve()
                for path in sequence_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=frame_sort_key,
        )
        if expected_frames > 0 and len(sequence_frames) != expected_frames:
            bad_sequences.append((sequence_dir, len(sequence_frames)))
            if not allow_incomplete:
                continue
        frames.extend(sequence_frames)

    if bad_sequences and not allow_incomplete:
        preview = "\n".join(
            f"{folder}: {count} frames" for folder, count in bad_sequences[:20]
        )
        more = "" if len(bad_sequences) <= 20 else f"\n... {len(bad_sequences) - 20} more"
        raise RuntimeError(
            f"Some sequences do not have expected_frames={expected_frames}:\n"
            f"{preview}{more}\n"
            "Use --allow-incomplete true if you still want to test them."
        )
    if not frames:
        raise RuntimeError("No test frames were collected.")

    return frames, sequence_dirs, bad_sequences


def write_temp_filelist(frames):
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_uf_test_filelist.txt",
        delete=False,
        encoding="utf-8",
    )
    with temp_file:
        for path in frames:
            temp_file.write(str(path) + "\n")
    return temp_file.name


def bitstream_size(bit_stream):
    return len(bit_stream)


def write_bitstream(path, bit_stream):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bit_stream)


def save_recon_images(x_hat, output_dir, chunk_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = ycbcr2rgb(x_hat[0].detach().cpu(), clamp=True)
    rgb = (rgb.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    rgb = rgb.permute(0, 2, 3, 1).contiguous().numpy()

    for frame_idx, frame in enumerate(rgb):
        imageio.imwrite(output_dir / f"chunk_{chunk_index:06d}_frame_{frame_idx:02d}.png", frame)


def build_test_loader(args):
    temp_filelist = None
    source_info = None
    root = args.test_dataset
    filelist = args.test_filelist

    if filelist is None:
        frames, sequence_dirs, bad_sequences = collect_test_frames(
            args.test_dataset,
            args.test_classes,
            args.expected_frames,
            args.allow_incomplete,
        )
        temp_filelist = write_temp_filelist(frames)
        root = None
        filelist = temp_filelist
        source_info = {
            "sequence_count": len(sequence_dirs),
            "frame_count": len(frames),
            "bad_sequence_count": len(bad_sequences),
            "classes": args.test_classes,
        }

    try:
        dataset = UFTestDataSet(
            root=root,
            filelist=filelist,
            gop=args.gop,
            chunk_size=args.chunk_size,
            testfull=args.testfull,
            align=args.align,
            pad_last=args.pad_last,
        )
    finally:
        if temp_filelist is not None:
            Path(temp_filelist).unlink(missing_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    return dataset, dataloader, source_info


def merge_gop_chunks(batch):
    ref_chunk = batch[0]
    input_chunks = batch[1]
    if torch.is_tensor(input_chunks) and input_chunks.numel() > 0:
        chunks = torch.cat([ref_chunk.unsqueeze(1), input_chunks], dim=1)
        b, chunk_num, t, c, h, w = chunks.shape
        return chunks.reshape(b * chunk_num, t, c, h, w)
    return ref_chunk


def should_save_recon(chunk_index, args):
    if not args.save_recon:
        return False
    if args.save_recon_chunks < 0:
        return True
    return chunk_index < args.save_recon_chunks


def test_one_qp(model, dataloader, qp, checkpoint_output_dir, args):
    model.eval()
    meters = {
        "bpp": AverageMeter(),
        "mse": AverageMeter(),
        "psnr_yuv420": AverageMeter(),
        "bin_bytes": AverageMeter(),
        "enc_ms": AverageMeter(),
        "dec_ms": AverageMeter(),
    }
    start = time.time()
    chunk_index = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            chunks = merge_gop_chunks(batch).to(args.device, non_blocking=True)
            for item_idx in range(chunks.size(0)):
                ref_chunk = chunks[item_idx:item_idx + 1]
                padded_ref_chunk, orig_h, orig_w = pad_chunk_to_multiple(
                    ref_chunk,
                    args.model_align,
                )
                _, t, _, padded_h, padded_w = padded_ref_chunk.shape

                maybe_synchronize(args.device)
                enc_start = time.perf_counter()
                out_enc = model.compress(padded_ref_chunk, qp)
                maybe_synchronize(args.device)
                enc_ms = (time.perf_counter() - enc_start) * 1000.0

                bit_stream = out_enc["bit_stream"]
                bin_bytes = bitstream_size(bit_stream)
                if args.save_bin:
                    bin_path = checkpoint_output_dir / "bin" / f"qp_{qp}" / f"chunk_{chunk_index:06d}.bin"
                    write_bitstream(bin_path, bit_stream)

                sps = {"height": padded_h, "width": padded_w, "ec_part": 0}
                maybe_synchronize(args.device)
                dec_start = time.perf_counter()
                out_dec = model.decompress(bit_stream, sps, qp)
                maybe_synchronize(args.device)
                dec_ms = (time.perf_counter() - dec_start) * 1000.0

                x_hat = crop_chunk(out_dec["x_hat"], orig_h, orig_w)
                mse = chunk_mse(ref_chunk, x_hat)
                bpp = bin_bytes * 8.0 / float(t * orig_h * orig_w)

                meters["bpp"].update(bpp)
                meters["mse"].update(mse)
                meters["psnr_yuv420"].update(yuv420_weighted_psnr(ref_chunk, x_hat))
                meters["bin_bytes"].update(bin_bytes)
                meters["enc_ms"].update(enc_ms)
                meters["dec_ms"].update(dec_ms)

                if should_save_recon(chunk_index, args):
                    recon_dir = checkpoint_output_dir / "recon" / f"qp_{qp}"
                    save_recon_images(x_hat, recon_dir, chunk_index)

                chunk_index += 1

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                print(
                    "QP {:2d} [{:4d}/{:4d}] "
                    "PSNR {:.3f} | YUV420-PSNR {:.3f} | MSE {:.8f} | "
                    "BPP {:.4f} | Bin {:.2f} B | Enc {:.2f} ms | Dec {:.2f} ms".format(
                        qp,
                        batch_idx + 1,
                        len(dataloader),
                        psnr_from_mse(meters["mse"].avg),
                        meters["psnr_yuv420"].avg,
                        meters["mse"].avg,
                        meters["bpp"].avg,
                        meters["bin_bytes"].avg,
                        meters["enc_ms"].avg,
                        meters["dec_ms"].avg,
                    )
                )

            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break

    return {
        "qp": qp,
        "bpp": meters["bpp"].avg,
        "mse": meters["mse"].avg,
        "psnr": psnr_from_mse(meters["mse"].avg),
        "psnr_yuv420": meters["psnr_yuv420"].avg,
        "bin_bytes": meters["bin_bytes"].avg,
        "enc_ms": meters["enc_ms"].avg,
        "dec_ms": meters["dec_ms"].avg,
        "chunks": meters["mse"].count,
        "time": time.time() - start,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Test trained DCVCUFIntra image model on 8-frame chunks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The single phase-1 checkpoint to test.",
    )
    parser.add_argument(
        "-td",
        "--test-dataset",
        type=str,
        default="/home/admin1/Data/data/vimeo_septuplet",
        help="Test root used when paths in --test-filelist are relative.",
    )
    parser.add_argument(
        "-td_l",
        "--test-filelist",
        type=str,
        default="/home/admin1/Data/data/vimeo_septuplet/test_filelist.txt",
        help="Text file containing the test frame paths.",
    )
    parser.add_argument("--test-classes", nargs="*", default=DEFAULT_TEST_CLASSES)
    parser.add_argument("--expected-frames", type=int, default=96)
    parser.add_argument("--allow-incomplete", type=str2bool, default=False)
    parser.add_argument("--qps", type=int, nargs="+", default=[20])
    parser.add_argument("--q-num", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument("--align", type=int, default=8)
    parser.add_argument(
        "--model-align",
        type=int,
        default=16,
        help="Pad input H/W to this multiple before running the model, then crop metrics back.",
    )
    parser.add_argument("--testfull", type=str2bool, default=True)
    parser.add_argument("--pad-last", type=str2bool, default=True)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--disable-custom-cuda",
        type=str2bool,
        default=False,
        help="Disable customized CUDA inference kernels and use PyTorch fallback layers.",
    )
    parser.add_argument(
        "--disable-depthconv-proxy",
        type=str2bool,
        default=True,
        help="Disable DepthConvProxy while keeping other customized CUDA inference kernels enabled.",
    )
    parser.add_argument(
        "--disable-subpel-proxy",
        type=str2bool,
        default=True,
        help="Disable SubpelConv2xProxy while keeping other customized CUDA inference kernels enabled.",
    )
    parser.add_argument(
        "--disable-round-int8-cuda",
        type=str2bool,
        default=True,
        help="Disable round_and_to_int8_cuda while keeping other customized CUDA inference kernels enabled.",
    )
    parser.add_argument(
        "--disable-clamp-reciprocal-cuda",
        type=str2bool,
        default=True,
        help="Disable clamp_reciprocal_with_quant_cuda while keeping other customized CUDA inference kernels enabled.",
    )
    parser.add_argument(
        "--disable-build-index-cuda",
        type=str2bool,
        default=True,
        help="Disable build_index_dec/enc CUDA kernels while keeping other customized CUDA inference kernels enabled.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to save console test output. Default: test_uf/test_phase_1_*.log",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for .bin files and reconstructed images. Default: test_uf/codec_outputs/<log_name>",
    )
    parser.add_argument("--save-bin", type=str2bool, default=True)
    parser.add_argument("--save-recon", type=str2bool, default=True)
    parser.add_argument(
        "--save-recon-chunks",
        type=int,
        default=1,
        help="Number of chunks per checkpoint/QP to save as PNG. Use -1 to save all.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    if args.device is None:
        args.device = torch.device(
            "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        )
    else:
        args.device = torch.device(args.device)

    if args.chunk_size != 8:
        raise ValueError("DCVCUFIntra phase1 image model requires chunk_size=8.")
    if args.gop < args.chunk_size:
        raise ValueError("gop must be >= chunk_size.")
    if args.model_align < args.align or args.model_align % args.align != 0:
        raise ValueError("model_align must be a multiple of align and >= align.")
    if not (0 <= min(args.qps) and max(args.qps) < args.q_num <= 64):
        raise ValueError("All qps must be in [0, q_num), and q_num must be <= 64.")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.test_filelist is not None and not Path(args.test_filelist).is_file():
        raise FileNotFoundError(f"Test filelist not found: {args.test_filelist}")
    if args.test_dataset is not None and not Path(args.test_dataset).exists():
        raise FileNotFoundError(f"Test dataset root not found: {args.test_dataset}")
    if args.device.type != "cuda":
        raise RuntimeError("Real compress/decompress test requires CUDA. Set CUDA_VISIBLE_DEVICES to an idle GPU.")
    return args


def main(argv):
    args = parse_args(argv)
    if args.disable_custom_cuda:
        disable_custom_cuda_inference()
    set_custom_cuda_kernel_enabled(
        not args.disable_depthconv_proxy,
        not args.disable_subpel_proxy,
        not args.disable_round_int8_cuda,
        not args.disable_clamp_reciprocal_cuda,
        not args.disable_build_index_cuda,
    )
    torch.backends.cudnn.benchmark = args.device.type == "cuda"

    log_path, log_handle, original_stdout = setup_stdout_log(
        args.log_file if args.log_file is not None else default_log_path()
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else default_output_dir(log_path).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint_path = Path(args.checkpoint).resolve()
        label = checkpoint_label(checkpoint_path, REPO_ROOT)
        checkpoint_output_dir = output_dir / sanitize_path_part(label)
        print(f"Log file: {log_path}")
        print(f"Output dir: {output_dir}")
        print(f"Fixed QP(s): {', '.join(str(qp) for qp in args.qps)}")
        print(f"Save bin: {args.save_bin}")
        print(f"Save recon: {args.save_recon} | save_recon_chunks: {args.save_recon_chunks}")
        print(f"Device: {args.device}")
        print(f"Custom CUDA inference: {not args.disable_custom_cuda}")
        print(f"DepthConvProxy: {not args.disable_custom_cuda and not args.disable_depthconv_proxy}")
        print(f"SubpelConv2xProxy: {not args.disable_custom_cuda and not args.disable_subpel_proxy}")
        print(f"round_and_to_int8_cuda: {not args.disable_custom_cuda and not args.disable_round_int8_cuda}")
        print(
            "clamp_reciprocal_with_quant_cuda: "
            f"{not args.disable_custom_cuda and not args.disable_clamp_reciprocal_cuda}"
        )
        print(
            "build_index_dec/enc_cuda: "
            f"{not args.disable_custom_cuda and not args.disable_build_index_cuda}"
        )
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test dataset: {args.test_dataset}")
        print(f"Test filelist: {args.test_filelist}")

        dataset, dataloader, source_info = build_test_loader(args)
        if source_info is not None:
            print(
                "Collected {} frames from {} sequences".format(
                    source_info["frame_count"],
                    source_info["sequence_count"],
                )
            )
            if source_info["bad_sequence_count"] > 0:
                print(f"Sequences with non-standard frame count: {source_info['bad_sequence_count']}")
        print(
            "Loaded {} GOP samples, gop={}, chunk_size={}".format(
                len(dataset), args.gop, args.chunk_size
            )
        )

        print("\n" + "=" * 80)
        print(f"Testing checkpoint: {label}")
        print(f"Checkpoint output dir: {checkpoint_output_dir}")

        model = DCVCUFIntra()
        checkpoint = load_checkpoint(model, checkpoint_path)
        model = model.to(args.device)
        model.eval()

        checkpoint_epoch = checkpoint.get("epoch", None)
        checkpoint_epoch = int(checkpoint_epoch) if checkpoint_epoch is not None else -1
        print(f"checkpoint_epoch: {checkpoint_epoch}")

        checkpoint_results = []
        for qp in args.qps:
            result = test_one_qp(model, dataloader, qp, checkpoint_output_dir, args)
            result["checkpoint"] = label
            result["checkpoint_path"] = str(checkpoint_path)
            result["checkpoint_epoch"] = checkpoint_epoch
            checkpoint_results.append(result)

        print("\nTest Summary")
        print("QP | Chunks | PSNR | YUV420-PSNR | MSE | BPP | Bin(B) | Enc(ms) | Dec(ms) | Time(s)")
        for result in checkpoint_results:
            print(
                "{qp:2d} | {chunks:6d} | {psnr:.3f} | {psnr_yuv420:.3f} | "
                "{mse:.8f} | {bpp:.4f} | {bin_bytes:.2f} | "
                "{enc_ms:.2f} | {dec_ms:.2f} | {time:.2f}".format(
                    **result
                )
            )

        del model
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"\nSaved outputs to: {output_dir}")
        print(f"\nSaved log to: {log_path}")
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main(sys.argv[1:])
