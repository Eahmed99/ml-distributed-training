from pathlib import Path
from torchvision import datasets, transforms

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

if __name__ == "__main__":
    train_dataset, test_dataset = load_cifar10()

    image, label = train_dataset[0]

    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Image pixel range:", image.min().item(), image.max().item())
    print("Label:", label)
    print("Class:", train_dataset.classes[label])
