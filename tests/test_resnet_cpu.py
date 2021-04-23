import pytest

import torch

import sys
sys.path.insert(0, 'resnet-pytorch')

from resnet import resnet20
from utils import select_device, count_layers


def test_device_is_cpu():
    global device
    device = select_device()
    assert 'cpu' == device


def test_resnet20():
    model = resnet20()
    model = model.to(device)
    print(model)
    print(count_layers(model))
    assert False == next(model.parameters()).is_cuda

    inputs = torch.randn((1, 3, 32, 32))
    inputs = inputs.to(device)
    assert False == inputs.is_cuda

    with torch.no_grad():
        outputs = model(inputs)
        assert False == outputs.is_cuda
        assert outputs.shape[0] == inputs.shape[0]
