import torch
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from mlso.data import load_cifar10, create_train_loader
from mlso.model import create_model
import torch.nn as nn
from mlso.train import train_one_epoch


def create_distributed_model(local_rank):

    device = torch.device("cuda", local_rank)
    model = create_model()
    model = model.to(device)

    model = DDP(

        model,
        device_ids= [local_rank],
        output_device= local_rank,
    )

    return model


def distributed_worker():

    try:
        # Initilaize

        rank, world_size, local_rank = setup_distributed()

        # Idenitfy worker gpu

        device = torch.device("cuda", local_rank)

        # create ddp wrapper

        model = create_distributed_model(local_rank)

        # Loading and dividing the data between gpu

        train_dataset, test_dataset = load_cifar10()

        train_sampler = DistributedSampler(

            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
            drop_last= False,
        )

        local_batch_size = 128 // world_size # maontiang gloabl batch size of 128


        # Create the worker

        train_loader = create_train_loader(
            train_dataset,
            batch_size= local_batch_size,
            sampler= train_sampler,
        )

        # Trining loop inside each worker

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr =0.01
        )

        num_epoch = 1

        for epoch in range(num_epoch):

            train_sampler.set_epoch(epoch)

            average_loss, epoch_time = train_one_epoch(

                model,
                train_loader,
                criterion,
                optimizer,
                device,
                max_batches=3,
            )

            print(
                f"Rank {rank} | Epoch {epoch + 1} "
                f"| Local loss: {average_loss:.4f} "
                f"| Time: {epoch_time:.2f}s",
                flush=True,
            )

        print(
            f"Rank {rank}: "
            f"Dataset partition = {len(train_sampler)} images, "
            f"Local batch size = {local_batch_size}, "
            f"Batches = {len(train_loader)}",
            flush=True,
        )
        

        print(
            f"Worker {rank}/{world_size} "
            f"Using GPU {local_rank}"
        )
    finally:
        cleanup_distributed()

def setup_distributed():

    dist.init_process_group(

        backend= "nccl",
        init_method= "env://",

    )

    rank = dist.get_rank()

    world_size = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def cleanup_distributed():

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    distributed_worker()
