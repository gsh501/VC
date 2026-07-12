import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def default_sequences_dir():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Partvimeo_7" / "sequences"


def find_frame(folder, index):
    for ext in IMAGE_EXTENSIONS:
        path = folder / f"im{index}{ext}"
        if path.exists():
            return path
    return None


def expand_7_to_8(sequences_dir, overwrite=False):
    sequences_dir = Path(sequences_dir).resolve()
    copied = 0
    skipped = 0
    missing = 0

    for im7_path in sorted(sequences_dir.rglob("im7.*")):
        if im7_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        folder = im7_path.parent
        im8_path = folder / f"im8{im7_path.suffix}"

        existing_im8 = find_frame(folder, 8)
        if existing_im8 is not None and not overwrite:
            skipped += 1
            continue

        if not im7_path.exists():
            missing += 1
            continue

        if existing_im8 is not None and overwrite and existing_im8 != im8_path:
            existing_im8.unlink()

        shutil.copy2(im7_path, im8_path)
        copied += 1

    return copied, skipped, missing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy im7.* to im8.* for Vimeo-style 7-frame folders."
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=default_sequences_dir(),
        help="Root directory that contains Vimeo-style sequence folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing im8.* files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    copied, skipped, missing = expand_7_to_8(args.sequences_dir, args.overwrite)
    print(f"sequences_dir: {Path(args.sequences_dir).resolve()}")
    print(f"copied: {copied}")
    print(f"skipped_existing: {skipped}")
    print(f"missing_im7: {missing}")


if __name__ == "__main__":
    main()
