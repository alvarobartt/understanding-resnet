"""
Compare the inference time of the trained ResNet20 over the test set of CIFAR10 
using contiguous memory and channels last.

https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
"""

import os

from time import time

import math

import torch

from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from resnet import ResNet


def load_pretrained_resnet20():
    # Define global variable for the device
    global device

    # Check that GPU support is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define global variable for the model
    global model

    # Initiliaze ResNet20 for CIFAR10 and move it to the GPU (CPU if not available)
    model = ResNet(blocks=[3, 3, 3], filters=[16, 32, 64], num_classes=10)

    # Load the weights from the latest wandb run
    model.load_state_dict(torch.load("wandb/latest-run/files/resnet20-cifar10.pth"))
    
    # Move the model to the GPU as it was trained on a GPU
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-on-gpu-load-on-gpu
    model.to(device)
    
    # Set model on eval mode
    model.eval();


def load_test_cifar10_dataset(batch_size: int):
    # Define the mean/std normalization values (https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662)
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std= (0.2470, 0.2435, 0.2616)

    # Initialize/Define test transformation
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Define global variable for the dataloader
    global test_dataloader

    # Load CIFAR10 test dataset (transform it too), and initialize dataloader
    test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def convert_model_to_channels_last():
    # Reference: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
    global model
    model = model.to(memory_format=torch.channels_last)


def test_inference(channels_last: bool = False):
    total_time = .0

    for inputs, _ in test_dataloader:
        inputs = inputs.to(device)
        if channels_last: inputs.contiguous(memory_format=torch.channels_last)

        start_time = time()
        with torch.no_grad():
            _ = model(inputs)
        total_time += (time() - start_time)
    
    return total_time


if __name__ == '__main__':
    start_time = time()
    load_pretrained_resnet20()
    print(f"Pre-trained ResNet20 model loaded in: {(time() - start_time):.3f}s")

    # Warmup model with a simple random inference
    x = torch.randn((1, 3, 32, 32)).to(device)
    with torch.no_grad():
        _ = model(x)

    BATCH_SIZE = 128

    start_time = time()
    load_test_cifar10_dataset(batch_size=BATCH_SIZE)
    print(f"CIFAR10 test DataLoader loaded in: {(time() - start_time):.3f}s")

    num_batches = math.ceil(len(test_dataloader.dataset)/BATCH_SIZE)

    total_time = test_inference(channels_last=False)
    print(f"Inference using contiguous memory allocation took: {total_time:.3f}s ({(total_time/num_batches):.4f}s/step)")

    start_time = time()
    convert_model_to_channels_last()
    print(f"Pre-trained ResNet20 model conversion to channels-last took: {(time() - start_time):.3f}s")

    total_time = test_inference(channels_last=True)
    print(f"Inference using channels-last memory allocation took: {total_time:.3f}s ({(total_time/num_batches):.4f}s/step)")
