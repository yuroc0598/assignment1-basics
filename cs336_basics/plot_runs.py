import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_run(path: Path):
    steps, wall_clocks, train_losses = [], [], []
    val_steps, val_wall_clocks, val_losses = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "metrics":
                continue
            if "train_loss" in record:
                steps.append(record["step"])
                wall_clocks.append(record["wall_clock"])
                train_losses.append(record["train_loss"])
            if "val_loss" in record:
                val_steps.append(record["step"])
                val_wall_clocks.append(record["wall_clock"])
                val_losses.append(record["val_loss"])
    return steps, wall_clocks, train_losses, val_steps, val_wall_clocks, val_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay loss curves from one or more run logs.")
    parser.add_argument("runs", nargs="+", type=Path, help="Paths to run .jsonl log files.")
    parser.add_argument("--out", type=Path, default=Path("loss_curves.png"))
    args = parser.parse_args()

    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(12, 5))
    for run_path in args.runs:
        steps, wall_clocks, train_losses, val_steps, val_wall_clocks, val_losses = _load_run(run_path)
        label = run_path.stem
        ax_step.plot(steps, train_losses, label=f"{label} (train)")
        ax_time.plot(wall_clocks, train_losses, label=f"{label} (train)")
        if val_losses:
            ax_step.plot(val_steps, val_losses, "--", label=f"{label} (val)")
            ax_time.plot(val_wall_clocks, val_losses, "--", label=f"{label} (val)")

    ax_step.set_xlabel("step")
    ax_step.set_ylabel("loss")
    ax_step.set_title("Loss vs. steps")
    ax_step.legend()

    ax_time.set_xlabel("wall-clock time (s)")
    ax_time.set_ylabel("loss")
    ax_time.set_title("Loss vs. wall-clock time")
    ax_time.legend()

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
