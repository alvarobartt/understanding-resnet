# understanding-resnet

[![arXiv](https://img.shields.io/badge/arXiv-1512.03385-b31b1b.svg?style=flat)](https://arxiv.org/abs/1512.03385)

__TL;DR__ In Residual Learning the layers are reformulated as learning residual functions with
reference to the layer inputs. These networks are easier to optimize, and can gain accuracy
from considerably increased depth. Along this repository not just an explanation is provided
but also the implementation of the original ResNet architecture written in PyTorch. 
Additionally, here you will also find some ResNets trained with CIFAR10, as proposed by the
authors; which are some of the smallest ResNets described in the original paper. And, also some
ported weights for the bigger ResNets trained with ImageNet.

---

## :notebook: Explanation

Answering to the question "_Is learning better networks as easy as stacking more layers?_", so it's 
not as straight forward since we run into the problem of vanishing/exploding gradients, which is usually
addressed with batch normalization, so that those networks start converging for SGD with backprop.

Anyway, when those networks start converging, the degradation problem araises; so that the accuracy
gets saturated and then degrades rapidly, due to an increase in the training error (not overfitting).

__The authors address the degradation problem introducing the concept of residual learning__, which introduces
the idea of the residual mapping, where that there's no expectation that the stacked layers will fit
the underlying mapping, but the residual one. So that they state that it should be easier to optimize
the residual mapping than the unreferenced one.

So on, the "_shortcut connections_" are the ones connecting the input of a stack of convolutional layers
and the last convolutional layer on the stack, via skipping the intermediate connections.

<p align="center">
  <img width="400" height="250" src="https://user-images.githubusercontent.com/36760800/110832871-142dff80-829c-11eb-9c13-01d417e535d2.png"/>
</p>

<p align="center">
  <i>Source: <a href="https://arxiv.org/abs/1512.03385">Figure 2: Residual learning: a building block (courtesy of Kaiming He et al.)</a></i>
</p>

Based on this idea of "_shortcut connections_" the authors proposed CNN architectures using 
their deep residual learning approach, __fundamented on the hypothesis that if multiple
non-linear layers can asymptotically approximate complicated functions, they could also
approximate residual functions__.

So that on a block of stacked layers instead of approximating the underlying mapping, we
are approximating the residual function defined as: `F(x) := H(x) - x`; so that the stack
of layers approximates `F(x) + x`, where both have the same dimensions. This means that if
the identity mappings are optimal, the solvers may drive the weights of the multiple 
non-linear layers towards zero to approach the identity mappings.

__"_Shortcut connections_" do not include extra complexity, besides the element-wise addition__,
that has so little computational cost, so that it can be not taken into consideration. 
This is also helpful towards comparing both approaches, plain CNNs versus deep residual 
learning networks, as both have the same amount trainable parameters, and the same 
computational complexity.

__Usually the residual learning is adopted every few stacked layers__, with the condition
that the input dimention and the last layer's dimension in a building block is the 
same; and that in order to see advantages, the residual function should involve at
least 2 convolutional layers (experiments with 1 convolutional layer showed no great
improvement/advantage over plain CNNs).

__The bottleneck block design__ is a modification of the original building blocks proposed previously,
so that each residual function instead of using a stack of 2 convolutional layers now it will use a stack
of 3, which have a kernel size of 1x1, 3x3, and 1x1, respectively. The 1x1 convolutions are the ones in
charge of first reducing and then increasing the dimensions, which is called the restoration process. So that
the 3x3 convolution is left as a bottleneck with smaller input and output dimensions.

<p align="center">
  <img width="550" height="250" src="https://user-images.githubusercontent.com/36760800/114392408-05878080-9b99-11eb-9999-f37b87cdc348.png"/>
</p>

<p align="center">
  <i>Source: <a href="https://arxiv.org/abs/1512.03385">Figure 5: A deeper residual function F for ImageNet. Left: a building block (on 56??56 feature maps) as in  ResNet-34. Right: a ???bottleneck??? building block for ResNet-50/101/152. (courtesy of Kaiming He et al.)</a></i>
</p>

The bottleneck building blocks are used in the bigger architectures of ResNet, as they reduce the 
number of parameters and the amount of matrix multiplications to be calculated, so that the depth of the
neural network can be increased, but using less parameters than with the original residual blocks.

The authors proved that deep residual learning improved the performance of other CNN 
architectures for the ImageNet problem such as VGG or GoogLeNet. In addition to this, 
they also proved that when using deep residual learning compared to plain CNNs, the 
training error was decreasing when adding more layers, which was paliating the degradation 
problem.

---

## :pushpin: Useful concepts

<details>
  <summary><b>Vanishing/Exploding Gradients</b></summary>
  
  This is a common error that usually happens in deep nets, when
  the gradient of the early layers get vanishingly small, as consequence of the product of gradients between 
  0 and 1 in the deeper layers, which results in a smaller gradient. This means that the small gradient is 
  backwards propagated so that when it gets to the early layers, the update of the weights is small as the
  gradient is really small. Then this makes the net harder to train as the weights in the early layers suffer
  small changes, which means that the global or local minimum of the loss function won't be reached. Exploding
  gradients is the opposite, which means that the gradients that get propagated backwards are huge, so that the 
  weight updates are not able to find the best weights, and so on, not to able to find the global or local minimum
  of the loss function.
</details>

<details>
  <summary><b>Degradation Problem</b></summary>
  
  Deep nets tend to suffer from the degradation problem when including more layers 
  (increasing the depth of the net), so that the accuracy decreases, but this is not due to overfitting. We should
  expect that if a shallower net gets a certain accuracy, a deeper one should do at least as well as the shallower
  counterpart. Anyway, if those extra layers we include in the shallower net to make it deeper are just identity 
  mappings the accuracy should be the same as one achieved with the shallower net. But this doesn't happen, as the
  degradation problem appears as multiple non-linear layers can't learn the identity mappings, so that the accuracy
  gets degradated.
</details>

<details>
  <summary><b>Batch Normalization (BN)</b></summary>
  
  It's a technique for improving the performance and stability of neural nets. BN 
  keeps the back propagated gradients from getting too big or too small by rescaling and recentering the value of 
  all the hidden units in a mini-batch. BN mainly affects the intermediate layers, not the early ones, making all 
  the hidden units on each layer to have the same mean and variance, reducing the effect of the covariant shift. 
  BN reduces the problem of the input values changing when updating the learnable parameters, so that those values 
  are more stable. We should expect the normalized hidden units have a mean of zero and variance of one, just like 
  the normalization performed to the input values, but what we just want is to have them in a standard scale, to 
  avoid sparsity, which results on the net training slower and having inbalanced gradients, causing instability. So 
  on, BN speeds up the training as it makes the optimization landscape significantly smoother, which results in a 
  stabilization of the gradients accross all the neurons, allowing faster training ([1805.11604](https://arxiv.org/pdf/1805.11604.pdf)); 
  due to this, BN also allows sub-optimal weight initialization, so that that it is less important, as we will get 
  the local minimum in a similar number of iterations. As previously mentioned, BN also acts as a regularizer, since 
  it computes the mean and variance per every neuron activation on each mini-batch so that it's including a random 
  noise to both the mean and the standard deviation, forcing the downstream hidden units not to rely that much on a 
  hidden unit. Note that increasing the mini-batch size results on a poor regularization effect. Even though BN acts 
  as a normalizer with some sort of randomness, usually the deep nets keep BN together with Dropout, as both 
  regularizers combined tend to provide better results ([1905.05928](https://arxiv.org/pdf/1905.05928.pdf)). More 
  information about BN available in [1502.03167](https://arxiv.org/pdf/1502.03167.pdf).
</details>

<details>
  <summary><b>Shortcut/Skip Connections</b></summary>
  
  These connections are formulated so as to solve the degradation problem so that we
  can create a deeper net from the shallower version of it, without degradating the training accuracy. These 
  connections skip one or more layers.
</details>

<details>
  <summary><b>Residual Learning</b></summary>
  
  Instead of expecting a stack of layers to directly fit a desired underlying mapping, 
  those layers fit the residual mapping. So that instead of using a stack of non-linear layers to learn the identity 
  mappings we include the shortcut/skip,connection that connects the input of that stack of layers to the output of 
  the last layer in the stack, so that we ensure that the deeper counterpart achieves at least the same accuracy as
  the shallower counterpart. Instead of fitting the desired mapping `H(x)`, we fit another mapping `F(x) := H(x) - x`,
  so that the original mapping can be recasted to `F(x) + x`; in the worst case where the desired mapping `F(x)` is 
  0 (let's assume we are using ReLU as the activation function), we will still keep the residual mapping `x`; so 
  that the training accuracy will be at least as good in the deeper net as in its shallower counterpart.
</details>

<details>
  <summary><b>ResNet Block as a Bottleneck design</b></summary>
  
  In order to reduce the computational complexity of the `BasicBlock` that uses
  stacks of 2 3x3 convolutions, the `BottleneckBlock` uses stacks of 3 1x1, 3x3, and 1x1 convolutions. So that the 1x1 
  convolutions are the ones in charge of reducing the dimension, and increasing it, respectively. The process of decreasing
  the input dimensions before the 3x3 convolution, and then increasing it, is called "restoration process". Using these
  blocks in the deeper architectures helps us reduce the computational complexity, we end up with more layers but less 
  parameters that if we were using `BasicBlock` instead.
</details>

<details>
  <summary><b>Kaiming He Weight Initialization</b></summary>
  
  Initializing the weights before training deep networks lets us somehow handle
  the vanishing/exploding gradient problem as we are setting some initial weights, which are not randomly generated with a
  mean of zero and a standard deviation of one. Even though the ResNet architecture is prepared for a bad weight initialization
  thanks to the BatchNormalization layers, we will still define the weight initialization function so as to make the net more 
  resistent to vanishing/exploding gradients. So in Xavier Glorot initialization we calculate the variance of the weights so that
  for the activation function (sigmoid or tanh) those are still centered around zero, but with a smaller variance than 1, in this
  case 1/n where n is the number of weights connected to a node from the previous layer. So that after generating the random weights
  we multiply each of them by the square root of 1/n. If we are using a ReLU as the activation function instead (which is the most
  common scenario), the value for the variance of the weights is 2/n instead of 1/n, so that the randomly initialized weights
  are multiplied by the square root of 2/n. Also note that n is what in practice we call `fan_in`, but sometimes we may either use
  `fan_out` (the number of weights going out of the node) or both as the square root of 2/fan_in+fan_out. And, the biases are initialized
  at zero. More information about Kaiming He Weight Initialization available in [1502.01852](https://arxiv.org/pdf/1502.01852.pdf).
</details>

---

## :question: Why ResNets work?

Here you have a curated list of useful videos you can watch while I summarize all the information in a clear and concise way, 
so as to understand why ResNets work and how are ResNets used in practice:

- [ResNets Explained - Henry AI Labs](https://www.youtube.com/watch?v=sAzL4XMke80)
- [Deep Residual Learning for Image Recognition (Paper Explained) - Yannic Kilcher](https://www.youtube.com/watch?v=GWt6Fu05voI)
- [ResNets - DeepLearning.AI](https://www.youtube.com/watch?v=ZILIbUvp5lk)
- [Why ResNets work? - DeepLearning.AI](https://www.youtube.com/watch?v=RYth6EbBUqM)

---

## :computer: Usage

### CIFAR10

First import the model, and load the pre-trained set of weights:

```python
import sys
sys.path.insert(0, 'resnet-pytorch')

from resnet import resnet20

model = resnet20(zero_padding=False, pretrained=True)
```

Then you can test the inference of the pre-trained model as it follows:

```python
import torch

x = torch.randn((1, 3, 32, 32)) # batch_size, channels, height, width
y = model(x)
```

### ImageNet

First import the model, and load the pre-trained set of weights:

```python
import sys
sys.path.insert(0, 'resnet-pytorch')

from resnet import resnet18

model = resnet18(zero_padding=False, pretrained=True)
```

Then you can test the inference of the pre-trained model as it follows:

```python
import torch

x = torch.randn((1, 3, 224, 224)) # batch_size, channels, height, width
y = model(x)
```

---

## :memo: Cite the authors

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

---

## :open_book: References

### :bookmark_tabs: From research papers

* [He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

### :books: From books/posts/etc.

* [Howard, J., Gugger, S., &amp; Chintala, S. (2020). Chapter 14. ResNets. In Deep learning for coders with fastai and PyTorch: AI applications without a PhD (pp. 441???458). O'Reilly Media, Inc.](https://www.amazon.es/Deep-Learning-Coders-Fastai-Pytorch/dp/1492045527)

* [Residual Network Explained - Papers With Code: The latest in Machine Learning](https://paperswithcode.com/method/resnet)

### :computer: From code

* [Official PyTorch implementation of ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

* [Unofficial PyTorch implementation of ResNet - Ross Wightman](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py)
