"""Porting ResNet Pre-Trained Weights

PyTorch image models, scripts, pretrained weights by Ross Wightman
https://github.com/rwightman/pytorch-image-models
https://download.pytorch.org/models/resnet18-5c106cde.pth
"""

from collections import OrderedDict

from torch.hub import load_state_dict_from_url

from resnet import resnet18


def port_resnet18_weights(url):
    """Ports the pre-trained weights for ResNet-18 trained on ImageNet."""
    original_state_dict = load_state_dict_from_url(url)
    custom_state_dict = OrderedDict([])

    for k, v in original_state_dict.items():
        if k.startswith('layer'): k = k.replace('layer', 'rl')
        if k.__contains__('downsample'): k = k.replace('downsample', 'shortcut')
        custom_state_dict[k] = v

    model = resnet18(pretrained=False)
    model.load_state_dict(custom_state_dict)


if __name__ == '__main__':
    port_resnet18_weights(url='https://download.pytorch.org/models/resnet18-5c106cde.pth')
