import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from src_uf.dataload_uf import UFTestDataSet
from src_uf.models_uf.image_model import DCVCUFIntra
from src_uf.models_uf.video_model import DCVCUF

from test_phase_1 import (
    AverageMeter,
    bitstream_size,
    checkpoint_label,
    chunk_mse,
    crop_chunk,
    maybe_synchronize,
    pad_chunk_to_multiple,
    psnr_from_mse,
    sanitize_path_part,
    save_recon_images,
    setup_stdout_log,
    str2bool,
    torch_load,
    write_bitstream,
    yuv420_weighted_psnr,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "pretrained_uf"
    / "DCVCUF"
    / "4"
    / "checkpoint_best_loss_uf_phase_2.pth.tar"
)
DEFAULT_VIDEO_CHECKPOINT = (
    REPO_ROOT
    / "pretrained_uf"
    / "DCVCUF"
    / "4"
    / "checkpoint_best_loss_uf_phase_2_video.pth.tar"
)
DEFAULT_INTRA_CHECKPOINT = (
    REPO_ROOT
    / "pretrained_uf"
    / "DCVCUF"
    / "4"
    / "checkpoint_best_loss_uf_phase_2_intra.pth.tar"
)
DEFAULT_TEST_DATASET = Path("/home/admin1/Data/data/data")
DEFAULT_TEST_FILELIST = REPO_ROOT / "datafiles" / "train_phase_2" / "test_filelist_phase2.txt"


def default_log_path():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"test_phase_2_{timestamp}.log"


def default_output_dir(log_path):
    return Path(__file__).resolve().parent / "codec_outputs" / Path(log_path).stem


def is_state_dict(value):
    return (
        isinstance(value, dict)
        and bool(value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def extract_state_dict(checkpoint, keys):
    if is_state_dict(checkpoint):
        return checkpoint
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in keys:
        value = checkpoint.get(key)
        if is_state_dict(value):
            return value
    return None


def load_state(model, state_dict):
    if state_dict is None:
        raise ValueError("state_dict is missing.")
    state_dict = dict(state_dict)
    consume_prefix_in_state_dict_if_present(state_dict, prefix="module.")
    model.load_state_dict(state_dict)


def load_checkpoint_state(path, keys):
    checkpoint = torch_load(path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint, keys)
    if state_dict is None:
        raise ValueError(f"No usable state_dict found in checkpoint: {path}")
    return checkpoint, state_dict


def load_phase2_models(intra_model, video_encoder, video_decoder, checkpoint_path, args):
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch_load(checkpoint_path, map_location="cpu")

    video_state = extract_state_dict(
        checkpoint,
        ("state_dict", "video_state_dict", "net", "model"),
    )
    if video_state is None:
        raise ValueError(
            f"No DCVCUF video state found in {checkpoint_path}. "
            "Use --video-checkpoint for split video weights."
        )

    if args.intra_checkpoint is not None:
        _, intra_state = load_checkpoint_state(
            args.intra_checkpoint,
            ("intra_state_dict", "state_dict", "net", "model"),
        )
    else:
        intra_state = extract_state_dict(
            checkpoint,
            ("intra_state_dict", "image_state_dict", "intra_net"),
        )
        if intra_state is None:
            raise ValueError(
                f"No DCVCUFIntra state found in {checkpoint_path}. "
                "Pass --intra-checkpoint when testing split phase-2 weights."
            )

    load_state(video_encoder, video_state)
    load_state(video_decoder, video_state)
    load_state(intra_model, intra_state)
    return checkpoint if isinstance(checkpoint, dict) else {}


def build_test_loader(args):
    root = args.test_dataset
    filelist = args.test_filelist

    if filelist is None:
        raise ValueError("Phase-2 testing requires --test-filelist.")

    dataset = UFTestDataSet(
        root=root,
        filelist=filelist,
        gop=args.gop,
        chunk_size=args.chunk_size,
        testfull=args.testfull,
        align=args.align,
        pad_last=args.pad_last,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    return dataset, dataloader, None


def should_save_recon(chunk_index, args):
    if not args.save_recon:
        return False
    if args.save_recon_chunks < 0:
        return True
    return chunk_index < args.save_recon_chunks


def update_meters(meters, src, x_hat, bin_bytes, enc_ms, dec_ms):
    _, frame_num, _, height, width = src.shape
    bpp = bin_bytes * 8.0 / float(frame_num * height * width)
    mse = chunk_mse(src, x_hat)

    meters["bpp"].update(bpp)
    meters["mse"].update(mse)
    meters["psnr_yuv420"].update(yuv420_weighted_psnr(src, x_hat))
    meters["bin_bytes"].update(bin_bytes)
    meters["enc_ms"].update(enc_ms)
    meters["dec_ms"].update(dec_ms)


def write_optional_bitstream(checkpoint_output_dir, qp, gop_index, chunk_index, chunk_type, bit_stream, args):
    if not args.save_bin:
        return
    bin_path = (
        checkpoint_output_dir
        / "bin"
        / f"qp_{qp}"
        / f"gop_{gop_index:06d}_chunk_{chunk_index:02d}_{chunk_type}.bin"
    )
    write_bitstream(bin_path, bit_stream)


def initialize_cuda_device(device):
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.empty(1, device=device)


def compress_decompress_intra(intra_model, src_chunk, qp, args):
    padded_chunk, orig_h, orig_w = pad_chunk_to_multiple(src_chunk, args.model_align)
    _, frame_num, _, padded_h, padded_w = padded_chunk.shape

    maybe_synchronize(args.device)
    enc_start = time.perf_counter()
    out_enc = intra_model.compress(padded_chunk, qp)
    maybe_synchronize(args.device)
    enc_ms = (time.perf_counter() - enc_start) * 1000.0

    bit_stream = out_enc["bit_stream"]
    sps = {"height": padded_h, "width": padded_w, "ec_part": 0}

    maybe_synchronize(args.device)
    dec_start = time.perf_counter()
    out_dec = intra_model.decompress(bit_stream, sps, qp)
    maybe_synchronize(args.device)
    dec_ms = (time.perf_counter() - dec_start) * 1000.0

    x_hat = crop_chunk(out_dec["x_hat"], orig_h, orig_w)
    return {
        "bit_stream": bit_stream,
        "x_hat": x_hat,
        "enc_ref_chunk": out_enc.get("ref_chunk", out_enc["x_hat"]),
        "dec_ref_chunk": out_dec.get("ref_chunk", out_dec["x_hat"]),
        "enc_ms": enc_ms,
        "dec_ms": dec_ms,
        "frame_num": frame_num,
    }


def compress_decompress_video(video_encoder, video_decoder, src_chunk, qp, args):
    padded_chunk, orig_h, orig_w = pad_chunk_to_multiple(src_chunk, args.model_align)
    _, frame_num, _, padded_h, padded_w = padded_chunk.shape

    maybe_synchronize(args.device)
    enc_start = time.perf_counter()
    out_enc = video_encoder.compress(padded_chunk, qp)
    maybe_synchronize(args.device)
    enc_ms = (time.perf_counter() - enc_start) * 1000.0

    bit_stream = out_enc["bit_stream"]
    sps = {"height": padded_h, "width": padded_w, "ec_part": 0}

    maybe_synchronize(args.device)
    dec_start = time.perf_counter()
    out_dec = video_decoder.decompress(bit_stream, sps, qp)
    maybe_synchronize(args.device)
    dec_ms = (time.perf_counter() - dec_start) * 1000.0

    x_hat = crop_chunk(out_dec["x_hat"], orig_h, orig_w)
    return {
        "bit_stream": bit_stream,
        "x_hat": x_hat,
        "enc_ms": enc_ms,
        "dec_ms": dec_ms,
        "frame_num": frame_num,
    }


def reset_video_state(video_encoder, video_decoder, enc_ref_chunk, dec_ref_chunk):
    video_encoder.clear_dpb()
    video_decoder.clear_dpb()
    video_encoder.set_curr_poc(0)
    video_decoder.set_curr_poc(0)
    video_encoder.add_ref_key_chunk(enc_ref_chunk)
    video_decoder.add_ref_key_chunk(dec_ref_chunk)


def test_one_qp(intra_model, video_encoder, video_decoder, dataloader, qp, checkpoint_output_dir, args):
    intra_model.eval()
    video_encoder.eval()
    video_decoder.eval()
    meters = {
        "bpp": AverageMeter(),
        "mse": AverageMeter(),
        "psnr_yuv420": AverageMeter(),
        "bin_bytes": AverageMeter(),
        "enc_ms": AverageMeter(),
        "dec_ms": AverageMeter(),
    }
    start = time.time()
    gop_index = 0
    global_chunk_index = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            ref_chunks = batch[0].to(args.device, non_blocking=True)
            input_chunks = batch[1].to(args.device, non_blocking=True)

            for item_idx in range(ref_chunks.size(0)):
                src_i = ref_chunks[item_idx:item_idx + 1]
                result_i = compress_decompress_intra(intra_model, src_i, qp, args)
                bit_stream = result_i["bit_stream"]
                write_optional_bitstream(
                    checkpoint_output_dir,
                    qp,
                    gop_index,
                    0,
                    "i",
                    bit_stream,
                    args,
                )
                update_meters(
                    meters,
                    src_i,
                    result_i["x_hat"],
                    bitstream_size(bit_stream),
                    result_i["enc_ms"],
                    result_i["dec_ms"],
                )
                if should_save_recon(global_chunk_index, args):
                    recon_dir = checkpoint_output_dir / "recon" / f"qp_{qp}"
                    save_recon_images(result_i["x_hat"], recon_dir, global_chunk_index)
                global_chunk_index += 1

                reset_video_state(
                    video_encoder,
                    video_decoder,
                    result_i["enc_ref_chunk"],
                    result_i["dec_ref_chunk"],
                )

                for p_idx in range(input_chunks.size(1)):
                    src_p = input_chunks[item_idx, p_idx:p_idx + 1]
                    result_p = compress_decompress_video(
                        video_encoder,
                        video_decoder,
                        src_p,
                        qp,
                        args,
                    )
                    bit_stream = result_p["bit_stream"]
                    write_optional_bitstream(
                        checkpoint_output_dir,
                        qp,
                        gop_index,
                        p_idx + 1,
                        f"p{p_idx + 1}",
                        bit_stream,
                        args,
                    )
                    update_meters(
                        meters,
                        src_p,
                        result_p["x_hat"],
                        bitstream_size(bit_stream),
                        result_p["enc_ms"],
                        result_p["dec_ms"],
                    )
                    if should_save_recon(global_chunk_index, args):
                        recon_dir = checkpoint_output_dir / "recon" / f"qp_{qp}"
                        save_recon_images(result_p["x_hat"], recon_dir, global_chunk_index)
                    global_chunk_index += 1

                gop_index += 1

            if args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                print(
                    "QP {:2d} [{:4d}/{:4d}] "
                    "GOPs {:5d} | Chunks {:6d} | PSNR {:.3f} | "
                    "YUV420-PSNR {:.3f} | MSE {:.8f} | BPP {:.4f} | "
                    "Bin {:.2f} B | Enc {:.2f} ms | Dec {:.2f} ms".format(
                        qp,
                        batch_idx + 1,
                        len(dataloader),
                        gop_index,
                        meters["mse"].count,
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
        "gops": gop_index,
        "chunks": meters["mse"].count,
        "time": time.time() - start,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Test phase-2 DCVCUF video model with DCVCUFIntra I chunks and DCVCUF P chunks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Combined phase-2 checkpoint. It must contain state_dict and intra_state_dict.",
    )
    parser.add_argument(
        "--video-checkpoint",
        type=str,
        default=None,
        help="Split DCVCUF video checkpoint. Requires --intra-checkpoint.",
    )
    parser.add_argument(
        "--intra-checkpoint",
        type=str,
        default=None,
        help="Split DCVCUFIntra checkpoint, or override intra weights from a combined checkpoint.",
    )
    parser.add_argument(
        "-td",
        "--test-dataset",
        type=str,
        default=str(DEFAULT_TEST_DATASET),
        help="Test root used when paths in --test-filelist are relative.",
    )
    parser.add_argument(
        "-td_l",
        "--test-filelist",
        type=str,
        default=str(DEFAULT_TEST_FILELIST),
        help="Text file containing the phase-2 test frame paths.",
    )
    parser.add_argument("--qps", type=int, nargs="+", default=[20])
    parser.add_argument("--q-num", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gop", type=int, default=32)
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
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to save console test output. Default: test_uf/test_phase_2_*.log",
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
        default=4,
        help="Number of chunks per checkpoint/QP to save as PNG. Use -1 to save all.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    if args.checkpoint is None and args.video_checkpoint is None:
        args.checkpoint = str(DEFAULT_CHECKPOINT)

    if args.device is None:
        args.device = torch.device(
            "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        )
    else:
        args.device = torch.device(args.device)

    if args.checkpoint is not None and args.video_checkpoint is not None:
        parser.error("Use either --checkpoint or --video-checkpoint, not both.")
    if args.video_checkpoint is not None and args.intra_checkpoint is None:
        parser.error("--video-checkpoint requires --intra-checkpoint.")
    if args.chunk_size != 8:
        raise ValueError("DCVCUF phase2 model requires chunk_size=8.")
    if args.gop < args.chunk_size or args.gop % args.chunk_size != 0:
        raise ValueError("gop must be a positive multiple of chunk_size.")
    if args.model_align < args.align or args.model_align % args.align != 0:
        raise ValueError("model_align must be a multiple of align and >= align.")
    if not (0 <= min(args.qps) and max(args.qps) < args.q_num <= 64):
        raise ValueError("All qps must be in [0, q_num), and q_num must be <= 64.")
    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.video_checkpoint is not None and not Path(args.video_checkpoint).is_file():
        raise FileNotFoundError(f"Video checkpoint not found: {args.video_checkpoint}")
    if args.intra_checkpoint is not None and not Path(args.intra_checkpoint).is_file():
        raise FileNotFoundError(f"Intra checkpoint not found: {args.intra_checkpoint}")
    if args.test_filelist is None or not Path(args.test_filelist).is_file():
        raise FileNotFoundError(f"Test filelist not found: {args.test_filelist}")
    if args.test_dataset is not None and not Path(args.test_dataset).exists():
        raise FileNotFoundError(f"Test dataset root not found: {args.test_dataset}")
    if args.device.type != "cuda":
        raise RuntimeError("Real phase-2 compress/decompress test requires CUDA.")
    return args


def main(argv):
    args = parse_args(argv)
    initialize_cuda_device(args.device)
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
        checkpoint_path = Path(
            args.video_checkpoint if args.video_checkpoint is not None else args.checkpoint
        ).resolve()
        label = checkpoint_label(checkpoint_path, REPO_ROOT)
        checkpoint_output_dir = output_dir / sanitize_path_part(label)

        print(f"Log file: {log_path}")
        print(f"Output dir: {output_dir}")
        print(f"Fixed QP(s): {', '.join(str(qp) for qp in args.qps)}")
        print(f"Save bin: {args.save_bin}")
        print(f"Save recon: {args.save_recon} | save_recon_chunks: {args.save_recon_chunks}")
        print(f"Device: {args.device}")
        if args.video_checkpoint is not None:
            print(f"Video checkpoint: {checkpoint_path}")
            print(f"Intra checkpoint: {Path(args.intra_checkpoint).resolve()}")
        else:
            print(f"Checkpoint: {checkpoint_path}")
            if args.intra_checkpoint is not None:
                print(f"Intra checkpoint override: {Path(args.intra_checkpoint).resolve()}")
        print(f"Test dataset: {args.test_dataset}")
        print(f"Test filelist: {args.test_filelist}")

        dataset, dataloader, _ = build_test_loader(args)
        print(
            "Loaded {} GOP samples, gop={}, chunk_size={}".format(
                len(dataset), args.gop, args.chunk_size
            )
        )

        print("\n" + "=" * 80)
        print(f"Testing checkpoint: {label}")
        print(f"Checkpoint output dir: {checkpoint_output_dir}")

        intra_model = DCVCUFIntra()
        video_encoder = DCVCUF()
        video_decoder = DCVCUF()
        checkpoint = load_phase2_models(
            intra_model,
            video_encoder,
            video_decoder,
            checkpoint_path,
            args,
        )

        intra_model = intra_model.to(args.device)
        video_encoder = video_encoder.to(args.device)
        video_decoder = video_decoder.to(args.device)
        intra_model.eval()
        video_encoder.eval()
        video_decoder.eval()
        intra_model.update()
        video_encoder.update()
        video_decoder.update()

        checkpoint_epoch = checkpoint.get("epoch", None)
        checkpoint_epoch = int(checkpoint_epoch) if checkpoint_epoch is not None else -1
        print(f"checkpoint_epoch: {checkpoint_epoch}")

        checkpoint_results = []
        for qp in args.qps:
            result = test_one_qp(
                intra_model,
                video_encoder,
                video_decoder,
                dataloader,
                qp,
                checkpoint_output_dir,
                args,
            )
            result["checkpoint"] = label
            result["checkpoint_path"] = str(checkpoint_path)
            result["checkpoint_epoch"] = checkpoint_epoch
            checkpoint_results.append(result)

        print("\nTest Summary")
        print("QP | GOPs | Chunks | PSNR | YUV420-PSNR | MSE | BPP | Bin(B) | Enc(ms) | Dec(ms) | Time(s)")
        for result in checkpoint_results:
            print(
                "{qp:2d} | {gops:5d} | {chunks:6d} | {psnr:.3f} | "
                "{psnr_yuv420:.3f} | {mse:.8f} | {bpp:.4f} | "
                "{bin_bytes:.2f} | {enc_ms:.2f} | {dec_ms:.2f} | {time:.2f}".format(
                    **result
                )
            )

        del intra_model, video_encoder, video_decoder
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"\nSaved outputs to: {output_dir}")
        print(f"\nSaved log to: {log_path}")
    finally:
        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main(sys.argv[1:])
