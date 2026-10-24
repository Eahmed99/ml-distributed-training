import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from mlso.data import load_cifar10, create_train_loader, create_test_loader
from mlso.evaluate import evaluate_model
from mlso.model import create_model
from mlso.train import parse_args, seed_everything, train_one_epoch, write_metrics


def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_distributed_model(local_rank):
    device = torch.device("cuda", local_rank)
    model = create_model().to(device)
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


def distributed_worker():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    try:
        if args.batch_size % world_size:
            raise ValueError("Global batch size must be divisible by number of workers")
        device = torch.device("cuda", local_rank)
        seed_everything(args.seed)
        model = create_distributed_model(local_rank)
        train_dataset, test_dataset = load_cifar10()
        sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True, seed=args.seed, drop_last=False)
        train_loader = create_train_loader(train_dataset,
                                           batch_size=args.batch_size // world_size,
                                           sampler=sampler, seed=args.seed)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        # Only rank zero evaluates the unwrapped model. Other ranks wait below.
        test_loader = create_test_loader(test_dataset, batch_size=args.batch_size) if rank == 0 else None
        output = Path(args.output_dir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        rows = []
        cumulative_training = 0.0
        cumulative_response = 0.0
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            dist.barrier()
            epoch_start = time.perf_counter()
            profiler = None
            if args.profile and epoch == 0:
                from torch.profiler import profile, ProfilerActivity
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                profiler.__enter__()
            try:
                loss_sum, num_images, batches, local_seconds = train_one_epoch(
                    model, train_loader, criterion, optimizer, device,
                    max_batches=args.max_batches, return_stats=True)
            finally:
                if profiler:
                    profiler.__exit__(None, None, None)
                    profiler.export_chrome_trace(str(output / f"gpu_{world_size}_rank{rank}_trace.json"))
            # Measure the longest worker time as synchronous job training time.
            measured = torch.tensor([loss_sum, float(num_images), float(local_seconds)],
                                    dtype=torch.float64, device=device)
            dist.all_reduce(measured[:2], op=dist.ReduceOp.SUM)
            dist.all_reduce(measured[2:], op=dist.ReduceOp.MAX)
            global_loss = (measured[0] / measured[1]).item()
            global_images = int(measured[1].item())
            train_seconds = measured[2].item()
            dist.barrier()
            if rank == 0:
                accuracy = evaluate_model(model.module, test_loader, device)
                torch.cuda.synchronize(device)
            else:
                accuracy = None
            dist.barrier()
            # Rank 0 measures epoch wall time; this includes test evaluation and barriers.
            response_seconds = time.perf_counter() - epoch_start
            if rank == 0:
                cumulative_training += train_seconds
                cumulative_response += response_seconds
                rows.append(dict(epoch=epoch+1, workers=world_size, device=f"gpu_{world_size}",
                                 batches=batches, images=global_images, train_loss=global_loss,
                                 test_accuracy=accuracy, training_seconds=train_seconds,
                                 response_seconds=response_seconds,
                                 total_training_seconds=cumulative_training,
                                 total_response_seconds=cumulative_response,
                                 images_per_second=global_images/train_seconds,
                                 global_batch_size=args.batch_size, seed=args.seed,
                                 cpu_threads=""))
                write_metrics(output / f"gpu_{world_size}_metrics.csv", rows)
                torch.save(dict(epoch=epoch+1, model_state_dict=model.module.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(),
                                config=vars(args)), output / f"gpu_{world_size}_latest.pt")
                print(f"gpu_{world_size}: epoch {epoch+1}, {batches} steps, "
                      f"{global_images} images, global loss {global_loss:.4f}, "
                      f"test accuracy {accuracy:.2f}%, train {train_seconds:.2f}s, "
                      f"epoch response {response_seconds:.2f}s", flush=True)
            dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    distributed_worker()
