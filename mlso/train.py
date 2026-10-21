from mlso.data import (
    load_cifar10,
    create_train_loader,
    create_test_loader,
)

import torch.nn as nn
import torch
import time
from mlso.evaluate import evaluate_model
from mlso.model import create_model
from pathlib import Path
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet-18 CPU training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches per epoch; omit for full training"
    )

    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches =None):
    # Training one epoch

    model.train()

    total_loss = 0.0
    total_images = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    for batch_number, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

        if max_batches is not None and batch_number + 1 >= max_batches:

            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    epoch_time = time.perf_counter() - start_time

    average_loss = total_loss / total_images

    return average_loss, epoch_time


def main ():
    args = parse_args()
    torch.set_num_threads(4)
    print("PyTorch CPU threads:", torch.get_num_threads())

    # Load The data

    train_dataset, test_dataset = load_cifar10()

    # Create Datalaoder

    train_loader = create_train_loader(train_dataset)
    test_loader = create_test_loader(test_dataset)

    # using cuda if availble

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("selected device is:", device)

    # Create the Model

    model = create_model()

    model = model.to(device)

    # Cross Entroupy loss

    criterion = nn.CrossEntropyLoss()

    # Learning Starts

    # Create the optimiser

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr =0.01,
    )

    num_epoch = args.epochs

    checkpoint_dir = Path("checkpoints")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # writing metrics in a CSV for later comaprriosn

    metrics_dir = Path("benchmarks")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / "training_metrics.csv"

    with open(metrics_file, "w", newline="") as file:

        writer = csv.DictWriter(

            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_accuracy",
                "epoch_time_seconds",
            ],
        )

        writer.writeheader()

    for epoch in range(num_epoch):

        average_loss, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_batches,

        )

        accuracy = evaluate_model(model, test_loader, device)

        # writing training info in a csv


        with open(metrics_file, "a", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "test_accuracy",
                    "epoch_time_seconds",
                ],
            )

            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": average_loss,
                "test_accuracy": accuracy,
                "epoch_time_seconds": epoch_time,
            })


        print("Epoch:", epoch + 1)
        print("Average loss:", average_loss)
        print("Test accuracy:", accuracy, "%")
        print("Training time:", epoch_time, "seconds")

        
        # Creating a dictionary checkpoint
        checkpoint = {

            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        print("Check Point saved:", checkpoint_path)

if __name__ == "__main__":
    main()
    
