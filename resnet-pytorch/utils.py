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
