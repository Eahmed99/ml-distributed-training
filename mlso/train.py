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

    num_epoch = 5

    for epoch in range(num_epoch):

        average_loss, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        accuracy = evaluate_model(model, test_loader, device)
        print("Epoch:", epoch + 1)
        print("Average loss:", average_loss)
        print("Test accuracy:", accuracy, "%")
        print("Training time:", epoch_time, "seconds")

if __name__ == "__main__":
    main()
    
