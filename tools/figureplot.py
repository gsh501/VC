#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
EPOCH_SUMMARY_RE = re.compile(
    r"\b(?P<split>Train|Val)\s+epoch\s+(?P<epoch>\d+):\s+(?P<metrics>.*)$",
    re.IGNORECASE,
)
METRIC_RE = re.compile(rf"\b(?P<name>Loss|PSNR|MSE|BPP):\s*(?P<value>{NUMBER})\b", re.IGNORECASE)
METRICS = ("loss", "psnr", "bpp", "mse")
METRIC_LABELS = {
    "loss": "Loss",
    "psnr": "PSNR",
    "bpp": "BPP",
    "mse": "MSE",
}
METRIC_COLORS = {
    "loss": {"train": "#2563eb", "val": "#dc2626"},
    "psnr": {"train": "#0f766e", "val": "#7c3aed"},
    "bpp": {"train": "#ea580c", "val": "#0891b2"},
    "mse": {"train": "#4f46e5", "val": "#be123c"},
}


def parse_epoch_metrics(log_path):
    data = {metric: {"train": {}, "val": {}} for metric in METRICS}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_SUMMARY_RE.search(line)
            if not match:
                continue

            split = match.group("split").lower()
            epoch = int(match.group("epoch"))
            for metric_match in METRIC_RE.finditer(match.group("metrics")):
                metric = metric_match.group("name").lower()
                if metric in data:
                    data[metric][split][epoch] = float(metric_match.group("value"))

    return {
        metric: {
            split: sorted(epoch_values.items())
            for split, epoch_values in split_values.items()
            if epoch_values
        }
        for metric, split_values in data.items()
        if any(split_values.values())
    }


def default_output_dir(log_path):
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "figures" / log_path.stem


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise SystemExit(
            "Failed to import matplotlib. Please run this script in an environment "
            "with a working matplotlib installation, for example:\n"
            "  /home/admin1/anaconda3/envs/dcvc/bin/python tools/figureplot.py <log_path>"
        ) from exc

    return plt, MaxNLocator


def plot_metric(metric, metric_data, output_path, title=None, split="both", log_y=False, dpi=160):
    plt, MaxNLocator = import_pyplot()

    selected = []
    if split in ("both", "train") and "train" in metric_data:
        selected.append(("train", metric_data["train"]))
    if split in ("both", "val") and "val" in metric_data:
        selected.append(("val", metric_data["val"]))

    if not selected:
        available = ", ".join(sorted(metric_data)) or "none"
        raise SystemExit(f"No data found for split '{split}'. Available splits: {available}")

    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    label = METRIC_LABELS[metric]
    markers = {"train": "o", "val": "s"}

    for name, points in selected:
        epochs = [epoch for epoch, _ in points]
        values = [value for _, value in points]
        ax.plot(
            epochs,
            values,
            linewidth=1.8,
            marker=markers[name],
            markersize=3.2,
            color=METRIC_COLORS[metric][name],
            label=f"{name.title()} {label}",
        )

    ax.set_title(title or f"{label} by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel(f"{label} (log scale)")
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_metrics(data, output_dir, metrics, split="both", log_y=False, dpi=160):
    output_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if metric not in data:
            continue
        output_path = output_dir / f"{metric}_by_epoch.png"
        plot_metric(
            metric=metric,
            metric_data=data[metric],
            output_path=output_path,
            split=split,
            log_y=log_y,
            dpi=dpi,
        )
        output_paths.append(output_path)

    return output_paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Plot epoch-level train/val metric curves from a DCVC-style training log."
    )
    parser.add_argument("log_path", type=Path, help="Path to the training log file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to '<project_root>/figures/<log_stem>'.",
    )
    parser.add_argument(
        "--metric",
        choices=("all", *METRICS),
        default="all",
        help="Which metric to draw. Defaults to all metrics.",
    )
    parser.add_argument(
        "--split",
        choices=("both", "train", "val"),
        default="both",
        help="Which split curve to draw.",
    )
    parser.add_argument("--log-y", action="store_true", help="Use logarithmic y axis for saved figures.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    log_path = args.log_path.expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file does not exist: {log_path}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(log_path)
    data = parse_epoch_metrics(log_path)
    if not data:
        raise SystemExit(
            "No epoch-level metrics were found. Expected lines like:\n"
            "  Train epoch 0: Loss: 149.168 | ...\n"
            "  Val epoch 0: Loss: 32.060 | ..."
        )

    metrics = METRICS if args.metric == "all" else (args.metric,)
    output_paths = plot_metrics(
        data=data,
        output_dir=output_dir,
        metrics=metrics,
        split=args.split,
        log_y=args.log_y,
        dpi=args.dpi,
    )
    if not output_paths:
        available = ", ".join(sorted(data)) or "none"
        raise SystemExit(f"No requested metric was found. Available metrics: {available}")

    counts = []
    for metric in metrics:
        if metric not in data:
            continue
        split_counts = ",".join(f"{split}={len(points)}" for split, points in sorted(data[metric].items()))
        counts.append(f"{metric}({split_counts})")
    print(f"Parsed epochs: {'; '.join(counts)}")
    print(f"Saved figures to: {output_dir}")
    for output_path in output_paths:
        print(f"  {output_path.name}")


if __name__ == "__main__":
    main()
