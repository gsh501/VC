import argparse
import re
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
FRAME_RE = re.compile(r"^im(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_sequences_dir():
    return repo_root() / "Partvimeo_7" / "sequences"


def default_output_path():
    return repo_root() / "Partvimeo_7" / "train_filelist.txt"


def frame_sort_key(path):
    match = FRAME_RE.match(path.name)
    frame_index = int(match.group(1)) if match else 0
    return path.parent.as_posix(), frame_index, path.name


def collect_frames(sequences_dir):
    sequences_dir = Path(sequences_dir).resolve()
    frames = []

    for path in sequences_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if FRAME_RE.match(path.name) is None:
            continue
        frames.append(path.resolve())

    return sorted(frames, key=frame_sort_key)


def count_incomplete_folders(frames, expected_frames):
    folders = {}
    for path in frames:
        folders.setdefault(path.parent, set()).add(path.name.lower())

    incomplete = []
    for folder, names in folders.items():
        missing = [
            f"im{idx}.png"
            for idx in range(1, expected_frames + 1)
            if not any(name.startswith(f"im{idx}.") for name in names)
        ]
        if missing:
            incomplete.append((folder, missing))

    return incomplete


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
        description="Generate a txt filelist for UFDataSet from Vimeo-style frames."
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=default_sequences_dir(),
        help="Root directory that contains sequence folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output txt file path.",
    )
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
    return parser.parse_args()


def main():
    args = parse_args()
    sequences_dir = Path(args.sequences_dir).resolve()
    frames = collect_frames(sequences_dir)
    relative_to = sequences_dir if args.relative else None

    write_filelist(frames, args.output, relative_to=relative_to)
    incomplete = count_incomplete_folders(frames, args.expected_frames)

    print(f"sequences_dir: {sequences_dir}")
    print(f"output: {Path(args.output).resolve()}")
    print(f"frames: {len(frames)}")
    print(f"incomplete_folders: {len(incomplete)}")
    if incomplete:
        for folder, missing in incomplete[:20]:
            print(f"missing in {folder}: {', '.join(missing)}")
        if len(incomplete) > 20:
            print(f"... {len(incomplete) - 20} more incomplete folders")


if __name__ == "__main__":
    main()
