"""Create comparison tables and convergence plots from benchmark CSV files."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows_for(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def measure_collectives(trace_path):
    """Sum GPU NCCL kernel durations. This is collective work, not non-overlapped stall time."""
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    hits = []
    for event in events:
        name = str(event.get("name", "")).lower()
        category = str(event.get("cat", "")).lower()
        if "nccl" in name and "allreduce" in name.replace("_", "").replace("-", ""):
            if event.get("ph") == "X" and "dur" in event:
                hits.append(float(event["dur"]))
    return sum(hits) / 1e6, len(hits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="benchmarks")
    args = parser.parse_args()
    root = Path(args.dir)
    files = [root / f"{name}_metrics.csv" for name in ("cpu", "gpu_1", "gpu_2")]
    data = {p.name.removesuffix("_metrics.csv"): rows_for(p) for p in files if p.exists()}
    if not data:
        raise SystemExit(f"No *_metrics.csv files found in {root}")
    print("Configuration | epochs | training seconds | epoch response seconds | final accuracy | images/s")
    for config, rows in data.items():
        final = rows[-1]
        total_training = float(final["total_training_seconds"])
        total_response = float(final["total_response_seconds"])
        print(f"{config:>13} | {len(rows):>6} | {total_training:>16.2f} | "
              f"{total_response:>22.2f} | {float(final['test_accuracy']):>13.2f}% | "
              f"{float(final['images_per_second']):>8.1f}")
    if "gpu_1" in data and "gpu_2" in data:
        a, b = data["gpu_1"], data["gpu_2"]
        if len(a) == len(b):
            t1 = float(a[-1]["total_training_seconds"])
            t2 = float(b[-1]["total_training_seconds"])
            speedup = t1 / t2
            print(f"GPU scaling speedup S(2): {speedup:.3f}x")
            print(f"GPU parallel efficiency E(2): {speedup / 2 * 100:.2f}%")
            print(f"GPU accuracy gap (1 GPU - 2 GPUs): "
                  f"{float(a[-1]['test_accuracy']) - float(b[-1]['test_accuracy']):+.3f} points")
        else:
            print("Different epoch counts: refusing to calculate scaling speedup.")
    if "cpu" in data and "gpu_2" in data:
        a, b = data["cpu"], data["gpu_2"]
        if len(a) == len(b):
            ratio = float(a[-1]["total_training_seconds"]) / float(b[-1]["total_training_seconds"])
            print(f"CPU-to-2-GPU acceleration: {ratio:.3f}x (NOT GPU parallel efficiency)")
    for metric, filename, ylabel in [
        ("train_loss", "training_loss.png", "Mean training loss"),
        ("test_accuracy", "accuracy_vs_training_time.png", "Test accuracy (%)"),
        ("training_seconds", "epoch_times.png", "Training seconds / epoch")]:
        plt.figure(figsize=(7, 4.5))
        for config, rows in data.items():
            if metric == "train_loss":
                x = [int(r["epoch"]) for r in rows]
                xlabel = "Epoch"
            elif metric == "training_seconds":
                x = [int(r["epoch"]) for r in rows]
                xlabel = "Epoch"
            else:
                x = [float(r["total_training_seconds"]) for r in rows]
                xlabel = "Cumulative training time (s)"
            plt.plot(x, [float(r[metric]) for r in rows], marker="o", label=config)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(root / filename, dpi=160)
        plt.close()
    for trace_file in sorted(root.glob("*_trace.json")):
        try:
            seconds, count = measure_collectives(trace_file)
            print(f"{trace_file.name}: {seconds:.4f}s summed NCCL all-reduce events; "
                  f"{count} events (GPU kernel durations can overlap compute)")
        except (ValueError, KeyError) as exc:
            print(f"{trace_file.name}: cannot parse trace: {exc}")
    print("NCCL event totals are NOT communication stalls or exact overhead fractions.")
    print("Profiler traces may also include collective calls not attributable to gradient buckets.")


if __name__ == "__main__":
    main()
