"""
Compare the inference time of the trained ResNet20 over the test set of CIFAR10 
using contiguous memory and channels last.

https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
"""

import os

from time import time

import torch

from resnet import ResNet


def load_pretrained_resnet20():
    # Check that GPU support is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initiliaze ResNet20 for CIFAR10 and move it to the GPU (CPU if not available)
    model = ResNet(blocks=[3, 3, 3], filters=[16, 32, 64], num_classes=10)

    # Load the weights from the latest wandb run
    model.load_state_dict(torch.load("wandb/latest-run/files/resnet20-cifar10.pth"))
    
    # Move the model to the GPU as it was trained on a GPU
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-on-gpu-load-on-gpu
    model.to(device)
    
    # Set model on eval mode
    model.eval();

    return model


def inference_over_contiguous_images():
    pass


def inference_over_channel_last_images():
    pass


if __name__ == '__main__':
    start_time = time()
    model = load_pretrained_resnet20()
    print(f"Pre-trained ResNet20 model loaded in: {time() - start_time}s")

    x = torch.randn((1024, 3, 32, 32)).to('cuda')
    print(f"A contiguous Tensor's stride (# jumps in all the dims) looks like {x.stride()} in memory")
    
    start_time = time()
    y = model(x)
    print(f"Contiguous Tensor's inference time is: {time() - start_time}s")
    del y
    
    # Reference: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
    start_time = time()
    model = model.to(memory_format=torch.channels_last)
    print(f"ResNet20 model converted to channels-last in: {time() - start_time}s")

    x = x.contiguous(memory_format=torch.channels_last)
    print(f"A channels-last Tensor's stride (# jumps in all the dims) looks like {x.stride()} in memory")

    start_time = time()
    y = model(x)
    print(f"Channels-last Tensor's inference time is: {time() - start_time}s")
