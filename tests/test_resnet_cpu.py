import pytest

import torch

import sys
sys.path.insert(0, 'resnet-pytorch')

from resnet import resnet20
from utils import select_device, count_layers


def test_resnet20():
    model = resnet20()
    model.to(memory_format=torch.channels_last)
    print(count_layers(model))
    assert False == next(model.parameters()).is_cuda

    inputs = torch.randn((1, 3, 32, 32))
    inputs = inputs.contiguous(memory_format=torch.channels_last)
    assert False == inputs.is_cuda

    with torch.no_grad():
        outputs = model(inputs)
        assert False == outputs.is_cuda
        assert outputs.shape[0] == inputs.shape[0]
