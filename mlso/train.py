import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from mlso.data import load_cifar10, create_train_loader, create_test_loader
from mlso.evaluate import evaluate_model
from mlso.model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 training")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="For smoke tests only. Omit to train every batch.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Global batch size, fixed across configurations.")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--profile", action="store_true",
                        help="Save a profiler trace of the first epoch.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("epochs, batch-size and max-batches (if provided) must be positive")
    return args


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    max_batches=None, return_stats=False):
    model.train()
    total_loss = 0.0
    total_images = 0
    batches = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        n = labels.size(0)
        total_loss += loss.item() * n
        total_images += n
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if return_stats:
        return total_loss, total_images, batches, elapsed
    return total_loss / total_images, elapsed


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    seed_everything(args.seed)
    train_dataset, test_dataset = load_cifar10()
    train_loader = create_train_loader(train_dataset, batch_size=args.batch_size, seed=args.seed)
    test_loader = create_test_loader(test_dataset, batch_size=args.batch_size)
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = "cpu" if args.device == "cpu" else "gpu_1"
    rows = []
    cumulative_training = 0.0
    cumulative_wall = 0.0
    print(f"Device: {device}; epochs: {args.epochs}; batch size: {args.batch_size}; train: {len(train_dataset)}; test: {len(test_dataset)}", flush=True)
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        profiler = None
        if args.profile and epoch == 0:
            from torch.profiler import profile, ProfilerActivity
            acts = [ProfilerActivity.CPU]
            if device.type == "cuda":
                acts.append(ProfilerActivity.CUDA)
            profiler = profile(activities=acts, record_shapes=False)
            profiler.__enter__()
        try:
            loss_sum, images, batches, train_s = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                max_batches=args.max_batches, return_stats=True)
        finally:
            if profiler:
                profiler.__exit__(None, None, None)
                profiler.export_chrome_trace(str(output / f"{config}_trace.json"))
        accuracy = evaluate_model(model, test_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - epoch_start
        cumulative_training += train_s
        cumulative_wall += wall_s
        rows.append(dict(epoch=epoch+1, workers=1, device=config, batches=batches,
                         images=images, train_loss=loss_sum/images,
                         test_accuracy=accuracy, training_seconds=train_s,
                         response_seconds=wall_s, total_training_seconds=cumulative_training,
                         total_response_seconds=cumulative_wall,
                         images_per_second=images/train_s, global_batch_size=args.batch_size,
                         seed=args.seed, cpu_threads=args.cpu_threads if args.device == "cpu" else ""))
        write_metrics(output / f"{config}_metrics.csv", rows)
        torch.save(dict(epoch=epoch+1, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), config=vars(args)),
                   output / f"{config}_latest.pt")
        print(f"{config}: epoch {epoch+1}, {batches} batches, {images} images, "
              f"loss {loss_sum/images:.4f}, accuracy {accuracy:.2f}%, "
              f"train {train_s:.2f}s, epoch response {wall_s:.2f}s", flush=True)


if __name__ == "__main__":
    main()
