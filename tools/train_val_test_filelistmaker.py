import argparse
import random
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
FRAME_RE = re.compile(r"^im(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_dataset_root():
    candidates = [
        repo_root() / "Partvimeo_7",
        Path("/home/admin1/Data/data/vimeo_septuplet"),
    ]
    for root in candidates:
        if (root / "sequences").exists():
            return root
    return candidates[0]


def default_sequences_dir():
    return default_dataset_root() / "sequences"


def default_output_dir():
    return default_dataset_root()


def frame_sort_key(path):
    match = FRAME_RE.match(path.name)
    frame_index = int(match.group(1)) if match else 0
    return frame_index, path.name


def sequence_sort_key(folder):
    return folder.as_posix()


def collect_sequence_frames(sequences_dir):
    sequences_dir = Path(sequences_dir).resolve()
    sequence_frames = {}

    for path in sequences_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if FRAME_RE.match(path.name) is None:
            continue
        sequence_frames.setdefault(path.parent.resolve(), []).append(path.resolve())

    for folder in sequence_frames:
        sequence_frames[folder] = sorted(sequence_frames[folder], key=frame_sort_key)

    return dict(sorted(sequence_frames.items(), key=lambda item: sequence_sort_key(item[0])))


def get_missing_frames(frames, expected_frames):
    names = {path.name.lower() for path in frames}
    missing = []
    for idx in range(1, expected_frames + 1):
        prefix = f"im{idx}"
        if not any(name.startswith(prefix + ".") for name in names):
            missing.append(f"im{idx}")
    return missing


def split_counts(total, ratios):
    ratio_sum = sum(ratios)
    normalized = [ratio / ratio_sum for ratio in ratios]
    raw_counts = [total * ratio for ratio in normalized]
    counts = [int(count) for count in raw_counts]

    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1

    positive_indices = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive_indices):
        for idx in positive_indices:
            if counts[idx] == 0:
                donor = max(
                    positive_indices,
                    key=lambda donor_idx: counts[donor_idx],
                )
                if counts[donor] <= 1:
                    break
                counts[donor] -= 1
                counts[idx] = 1

    return counts


def split_sequences(sequence_folders, train_ratio, val_ratio, test_ratio, seed):
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if sum(ratios) <= 0:
        raise ValueError("At least one split ratio must be positive.")

    folders = list(sequence_folders)
    rng = random.Random(seed)
    rng.shuffle(folders)

    train_count, val_count, test_count = split_counts(len(folders), ratios)
    train_end = train_count
    val_end = train_end + val_count

    splits = {
        "train": folders[:train_end],
        "val": folders[train_end:val_end],
        "test": folders[val_end:val_end + test_count],
    }
    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=sequence_sort_key)
    return splits


def flatten_frames(folders, sequence_frames):
    frames = []
    for folder in folders:
        frames.extend(sequence_frames[folder])
    return frames


def write_filelist(frames, output_path, relative_to=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for path in frames:
        if relative_to is not None:
            line = path.relative_to(relative_to).as_posix()
        else:
            line = str(path)
        lines.append(line)

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split Vimeo-style sequence folders into train/val/test filelists."
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=default_sequences_dir(),
        help="Root directory that contains sequence folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory for train_filelist.txt, val_filelist.txt and test_filelist.txt.",
    )
    parser.add_argument("--train-output", type=str, default="train_filelist.txt")
    parser.add_argument("--val-output", type=str, default="val_filelist.txt")
    parser.add_argument("--test-output", type=str, default="test_filelist.txt")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Write paths relative to --sequences-dir instead of absolute paths.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=8,
        help="Expected frame count in every leaf sequence folder.",
    )
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Exclude sequence folders missing frames in [1, expected_frames].",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sequences_dir = Path(args.sequences_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not sequences_dir.exists():
        raise FileNotFoundError(f"sequences_dir not found: {sequences_dir}")

    sequence_frames = collect_sequence_frames(sequences_dir)
    if not sequence_frames:
        raise RuntimeError(f"No image frames found under: {sequences_dir}")

    incomplete = []
    for folder, frames in sequence_frames.items():
        missing = get_missing_frames(frames, args.expected_frames)
        if missing:
            incomplete.append((folder, missing))

    if args.drop_incomplete:
        incomplete_folders = {folder for folder, _ in incomplete}
        sequence_frames = {
            folder: frames
            for folder, frames in sequence_frames.items()
            if folder not in incomplete_folders
        }

    sequence_folders = list(sequence_frames.keys())
    splits = split_sequences(
        sequence_folders,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    outputs = {
        "train": output_dir / args.train_output,
        "val": output_dir / args.val_output,
        "test": output_dir / args.test_output,
    }
    relative_to = sequences_dir if args.relative else None

    split_frames = {}
    for split_name, folders in splits.items():
        frames = flatten_frames(folders, sequence_frames)
        split_frames[split_name] = frames
        write_filelist(frames, outputs[split_name], relative_to=relative_to)

    total_frames = sum(len(frames) for frames in sequence_frames.values())
    print(f"sequences_dir: {sequences_dir}")
    print(f"output_dir: {output_dir}")
    print(f"seed: {args.seed}")
    print(f"sequences: {len(sequence_frames)}")
    print(f"frames: {total_frames}")
    print(f"incomplete_folders: {len(incomplete)}")
    if args.drop_incomplete:
        print("drop_incomplete: yes")

    for split_name in ("train", "val", "test"):
        print(
            f"{split_name}: "
            f"sequences={len(splits[split_name])}, "
            f"frames={len(split_frames[split_name])}, "
            f"output={outputs[split_name]}"
        )

    if incomplete:
        for folder, missing in incomplete[:20]:
            print(f"missing in {folder}: {', '.join(missing)}")
        if len(incomplete) > 20:
            print(f"... {len(incomplete) - 20} more incomplete folders")


if __name__ == "__main__":
    main()
