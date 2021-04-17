"""ResNet PyTorch Utils

Contains some functions that are useful towards the implementation
and/or usage of all the available ResNet v1 variants.

Some of this functions include:
- Porting the weights from timm
- Changing the memory format Contiguous / Channels Last
- Normalization values of CIFAR10
"""

from __future__ import absolute_import

from collections import OrderedDict

import torch

from torch.hub import load_state_dict_from_url

from resnet import ResNet
from resnet import resnet18, resnet34, resnet50

VARIANTS = {
    "resnet18": {
        "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "model": resnet18
    },
    "resnet34": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth",
        "model": resnet34
    },
    "resnet50": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
        "model": resnet50
    }
}

# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662
MEAN_NORMALIZATION = (0.4914, 0.4822, 0.4465)
STD_NORMALIZATION = (0.247, 0.2435, 0.2616)


def port_resnet_weights_from_timm(variant: str) -> ResNet:
    """Ports the pre-trained weights for any ResNet v1 model from timm.

    Example:
        >>> from utils import port_resnet_weights_from_timm
        >>> model = port_resnet_weights_from_timm(variant="resnet18")
        >>> import torch
        >>> torch.save(model.state_dict(), "resnet18-ported-imagenet.pth")
    
    Reference:
        PyTorch image models, scripts, pretrained weights by Ross Wightman @ rwightman on GitHub
        https://github.com/rwightman/pytorch-image-models
    """

    assert variant in VARIANTS.keys()

    try:
        url = VARIANTS[variant]['url']
        original_state_dict = load_state_dict_from_url(url)
    except Exception as e:
        raise Exception(f"state_dict could not be loaded from URL with exception: {e}")

    custom_state_dict = OrderedDict([])

    # The known replacements between Ross Wightman's implementation and mine are defined
    for k, v in original_state_dict.items():
        if k.startswith("layer"): k = k.replace("layer", "rl")
        if k.__contains__("downsample"): k = k.replace("downsample", "subsample")
        custom_state_dict[k] = v

    del original_state_dict

    try:
        model = VARIANTS[variant]['model']
        model = model(pretrained=False)
        model.load_state_dict(custom_state_dict)
    except Exception as e:
        raise Exception(f"state_dict could not be ported as it can't be loaded with exception: {e}")

    return model


def convert_model_to_channels_last():
    return None


def convert_inputs_to_channels_last():
    return None


def convert_model_to_contiguous():
    return None


def warmup_model():
    return None

# """
# Compare the inference time of the pretrained ResNet20 over the test set of CIFAR10 
# using both `contiguous` (default) and `channels-last` memory allocation formats.

# PyTorch Reference: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
# """

# import os

# from time import time

# import math

# import torch

# from torch.hub import load_state_dict_from_url

# from torch.utils.data import DataLoader

# from torchvision import transforms as T
# from torchvision.datasets import CIFAR10

# from resnet import resnet20


# def load_pretrained_resnet20():
#     # Define global variable for the device
#     global device

#     # Check that GPU support is available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Define global variable for the model
#     global model

#     # Initiliaze ResNet20 for CIFAR10 and move it to the GPU (CPU if not available)
#     model = resnet20()

#     # # Load the weights from the latest wandb run
#     # model.load_state_dict(torch.load("wandb/latest-run/files/resnet20-cifar10.pth"))
#     # # or load the weights from the latest GitHub Release
#     model.load_state_dict(load_state_dict_from_url("https://github.com/alvarobartt/understanding-resnet/releases/download/v0.1/resnet20-cifar10.pth"))
    
#     # Move the model to the GPU as it was trained on a GPU
#     # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-on-gpu-load-on-gpu
#     model.to(device)
    
#     # Set model on eval mode
#     model.eval();


# def load_test_cifar10_dataset(batch_size: int):
#     # Define the mean/std normalization values (https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662)
#     norm_mean = (0.4914, 0.4822, 0.4465)
#     norm_std= (0.2470, 0.2435, 0.2616)

#     # Initialize/Define test transformation
#     test_transform = T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean=norm_mean, std=norm_std)
#     ])

#     # Define global variable for the dataloader
#     global test_dataloader

#     # Load CIFAR10 test dataset (transform it too), and initialize dataloader
#     test_dataset = CIFAR10(root="data", train=False, download=True, transform=test_transform)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# def convert_model_to_channels_last():
#     # Reference: https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
#     global model
#     model = model.to(memory_format=torch.channels_last)


# def test_inference(channels_last: bool = False):
#     total_time = .0

#     with torch.no_grad():
#         for inputs, _ in test_dataloader:
#             inputs = inputs.to(device)
#             if channels_last: inputs.contiguous(memory_format=torch.channels_last)

#             start_time = time()
#             _ = model(inputs)
#             total_time += (time() - start_time)

#     return total_time


# if __name__ == '__main__':
#     start_time = time()
#     load_pretrained_resnet20()
#     print(f"Pre-trained ResNet20 model loaded in {(time() - start_time):.3f}s")

#     # Warmup model with a simple random inference
#     x = torch.randn((1, 3, 32, 32)).to(device)
#     with torch.no_grad(): _ = model(x)

#     BATCH_SIZE = 128

#     start_time = time()
#     load_test_cifar10_dataset(batch_size=BATCH_SIZE)
#     print(f"CIFAR10 test DataLoader loaded in {(time() - start_time):.3f}s, with a batch size of {BATCH_SIZE}")

#     num_batches = math.ceil(len(test_dataloader.dataset)/BATCH_SIZE)
#     print(f"Number of batches/steps is {num_batches}")

#     total_time = test_inference(channels_last=False)
#     print(f"Inference using contiguous memory allocation took: {total_time:.3f}s ({(total_time/num_batches):.4f}s/step)")

#     start_time = time()
#     convert_model_to_channels_last()
#     print(f"Pre-trained ResNet20 model conversion to channels-last took: {(time() - start_time):.3f}s")

#     total_time = test_inference(channels_last=True)
#     print(f"Inference using channels-last memory allocation took: {total_time:.3f}s ({(total_time/num_batches):.4f}s/step)")
