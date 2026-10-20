import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_model(num_class=10):
    model = resnet18(weights = None)

    model.conv1 = nn.Conv2d(
        in_channels= 3,
        out_channels= 64,
        kernel_size= 3,
        stride= 1,
        padding= 1,
        bias= False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(

        in_features= model.fc.in_features,
        out_features= num_class,
    )

    return model



if __name__ == "__main__":

    model = create_model()

    # Create 4 artificial CIFAR-10-shaped images.
    dummy_images = torch.randn(4, 3, 32, 32)

    # Evaluation mode and no gradient tracking for this test.
    model.eval()

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)
