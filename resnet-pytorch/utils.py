"""ResNet PyTorch Utils

Contains some functions that are useful towards the implementation
and/or usage of all the available ResNet v1 variants trained both on
ImageNet and CIFAR10.

Some of this functions include:
- Porting the weights from timm
- Changing the memory format Contiguous / Channels Last
- Normalization values of CIFAR10
"""

from __future__ import absolute_import

from typing import Tuple

from collections import OrderedDict

import torch
import torch.nn as nn

from torch import Tensor
from torch.hub import load_state_dict_from_url

from resnet import ResNet
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152

IMAGENET_VARIANTS = {
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
    },
    "resnet101": {
        "url": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
        "model": resnet101
    },
    "resnet152": {
        "url": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        "model": resnet152
    }
}

# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#gistcomment-2851662
MEAN_NORMALIZATION = (0.4914, 0.4822, 0.4465)
STD_NORMALIZATION = (0.247, 0.2435, 0.2616)


def port_resnet_weights(variant: str) -> ResNet:
    """Ports the pre-trained weights for any ResNet v1 model trained on ImageNet from timm and/or PyTorch.

    Example:
        >>> from utils import port_resnet_weights
        >>> model = port_resnet_weights(variant="resnet18")
        >>> import torch
        >>> torch.save(model.state_dict(), "resnet18-imagenet-ported.pth")
    
    References:
        [1] PyTorch image models, scripts, pretrained weights https://github.com/rwightman/pytorch-image-models
        [2] torchvision: Datasets, Transforms and Models specific to Computer Vision https://github.com/pytorch/vision 
    """
    assert variant in IMAGENET_VARIANTS.keys()

    try:
        url = IMAGENET_VARIANTS[variant]['url']
        original_state_dict = load_state_dict_from_url(url)
    except Exception as e:
        raise Exception(f"state_dict could not be loaded from URL with exception: {e}")

    custom_state_dict = OrderedDict([])

    # The known replacements between Ross Wightman's/PyTorch implementation and mine are defined
    for k, v in original_state_dict.items():
        if k.startswith("layer"): k = k.replace("layer", "rl")
        if k.__contains__("downsample"): k = k.replace("downsample", "subsample")
        custom_state_dict[k] = v

    del original_state_dict

    try:
        model = IMAGENET_VARIANTS[variant]['model']
        model = model(pretrained=False)
        model.load_state_dict(custom_state_dict)
    except Exception as e:
        raise Exception(f"state_dict could not be ported as it can't be loaded with exception: {e}")

    return model


def convert_model_to_channels_last(model: nn.Module) -> nn.Module:
    """Converts the model to channels last memory format.
    
    References:
        [1] https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
    """
    model = model.to(memory_format=torch.channels_last)
    return model


def convert_inputs_to_channels_last(inputs: Tensor) -> Tensor:
    """Converts the inputs to channels last memory format (assuming the inputs are contiguous by default)."""
    inputs = inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def convert_model_to_contiguous(model: nn.Module) -> nn.Module:
    """Converts the model to contiguous memory format."""
    model = model.to(memory_format=torch.contiguous_format)
    return model


def warmup_model(model: nn.Module, input_size: Tuple[int, int, int], batch_size: int, channels_last: bool = False) -> None:
    """Warms up the model before running evaluating the inference time."""
    assert batch_size > 0
    
    device = select_device()

    model = model.to(device)
    if channels_last: convert_model_to_channels_last(model=model)
    model.eval()
    
    inputs = torch.randn((batch_size,)+input_size)
    inputs = inputs.to(device)
    if channels_last: convert_inputs_to_channels_last(model=model)

    with torch.no_grad(): _ = model(inputs)


def count_trainable_parameters(model: nn.Module) -> int:
    """Counts the total number of trainable parameters of a net."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_layers(model: nn.Module) -> int:
    """Counts the total number of layers of a net."""
    return len(list(filter(lambda param: param.requires_grad and len(param.data.size()) > 1, model.parameters())))


def select_device() -> str:
    """Selects the device to use, either CPU (default) or CUDA/GPU; assuming just one GPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"
