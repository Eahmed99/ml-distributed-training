from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch

def load_cifar10(data_dir="data"):
    # download dataset in memeory

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root = data_dir,
        train= True,
        download= False,
        transform = transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    return train_dataset, test_dataset

def create_train_loader(train_dataset, batch_size = 128, seed=42, sampler = None):
    # create a dataloader for training

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle =(sampler is None),
        num_workers = 0,
        drop_last = False,
        generator=generator,
        sampler= sampler


    )

    return train_loader


def create_test_loader(test_dataset, batch_size=128):
    #Create a DataLoader for model evaluation

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return test_loader



if __name__ == "__main__":
    train_dataset, test_dataset = load_cifar10()

    train_loader = create_train_loader(train_dataset, batch_size=128)
    test_loader = create_test_loader(test_dataset, batch_size=128)

    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    print("Training batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(test_loader))

    print("Test batch shape:", images.shape)
    print("Test labels shape:", labels.shape)

