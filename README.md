# understanding-resnet

[![arXiv](https://img.shields.io/badge/arXiv-1512.03385-b31b1b.svg?style=flat)](https://arxiv.org/abs/1512.03385)

__TL;DR__ In Residual Learning the layers are reformulated as learning residual functions with
reference to the layer inputs. These networks are easier to optimize, and can gain accuracy
from considerably increased depth. Along this repository not just an explanation is provided
but also the implementation of the ResNet architecture written in PyTorch, MXNet and JAX. 
Additionally, here you will also find the ResNet20 trained with CIFAR10, as proposed by the
authors; which is the smallest ResNet described in the original paper.

## :crystal_ball: Future Tasks

* [ ] Split implementations in PyTorch, PyTorch Lightning, JAX

![image](https://user-images.githubusercontent.com/36760800/112517074-3180c480-8d98-11eb-8fc5-df825000890b.png)

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

Finally, before proceeding with the implementation of the ResNet-20 based on the experiments
Kaiming He et al. conducted for the CIFAR10 dataset; we will mention that before running
these experiments, the authors already proved that deep residual learning improved the 
performance of other CNN architectures for the ImageNet problem such as VGG or GoogLeNet.
In addition to this, they also proved that when using deep residual learning compared
to plain CNNs, the training error was decreasing when adding more layers, which was 
paliating the degradation problem.

## :pushpin: Useful concepts

* __Vanishing/Exploding Gradients:__ this is a common error that usually happens in deep nets, when
the gradient of the early layers get vanishingly small, as consequence of the product of gradients between 
0 and 1 in the deeper layers, which results in a smaller gradient. This means that the small gradient is 
backwards propagated so that when it gets to the early layers, the update of the weights is small as the
gradient is really small. Then this makes the net harder to train as the weights in the early layers suffer
small changes, which means that the global or local minimum of the loss function won't be reached. Exploding
gradients is the opposite, which means that the gradients that get propagated backwards are huge, so that the 
weight updates are not able to find the best weights, and so on, not to able to find the global or local minimum
of the loss function.

* __Degradation Problem__: deep nets tend to suffer from the degradation problem when including more layers 
(increasing the depth of the net), so that the accuracy decreases, but this is not due to overfitting. We should
expect that if a shallower net gets a certain accuracy, a deeper one should do at least as well as the shallower
counterpart. Anyway, if those extra layers we include in the shallower net to make it deeper are just identity 
mappings the accuracy should be the same as one achieved with the shallower net. But this doesn't happen, as the
degradation problem appears as multiple non-linear layers can't learn the identity mappings, so that the accuracy
gets degradated.

* __Batch Normalization (BN)__: it's a technique for improving the performance and stability of neural nets. BN 
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

* __Shortcut/Skip Connections__: these connections are formulated so as to solve the degradation problem so that we
can create a deeper net from the shallower version of it, without degradating the training accuracy. These 
connections skip one or more layers.

* __Residual Learning__: instead of expecting a stack of layers to directly fit a desired underlying mapping, 
those layers fit the residual mapping. So that instead of using a stack of non-linear layers to learn the identity 
mappings we include the shortcut/skip,connection that connects the input of that stack of layers to the output of 
the last layer in the stack, so that we ensure that the deeper counterpart achieves at least the same accuracy as
the shallower counterpart. Instead of fitting the desired mapping `H(x)`, we fit another mapping `F(x) := H(x) - x`,
so that the original mapping can be recasted to `F(x) + x`; in the worst case where the desired mapping `F(x)` is 
0 (let's assume we are using ReLU as the activation function), we will still keep the residual mapping `x`; so 
that the training accuracy will be at least as good in the deeper net as in its shallower counterpart.

* __Kaiming He Weight Initialization__: bla -> More information about Kaiming He Weight Initialization available in [1502.01852](https://arxiv.org/pdf/1502.01852.pdf).

* __ResNet Block as a Bottleneck__: bla

## :question: Why ResNets work?

Here you have a curated list of useful videos you can watch while I summarize all the information in a clear and concise way, 
so as to understand why ResNets work and how are ResNets used in practice:

- [ResNets Explained](https://www.youtube.com/watch?v=sAzL4XMke80)
- [Deep Residual Learning for Image Recognition (Paper Explained) - Yannic Kilcher](https://www.youtube.com/watch?v=GWt6Fu05voI)
- [ResNets - DeepLearning.AI](https://www.youtube.com/watch?v=ZILIbUvp5lk)
- [Why ResNets work? - DeepLearning.AI](https://www.youtube.com/watch?v=RYth6EbBUqM)

## :test_tube: ResNet-20 Implementation

In order to understand how does the ResNet architecture work, we will be implementing the simplest version of it, which
is the ResNet20 for CIFAR10. This exercise will be useful to understand the main differences between a plain convolutional
neural network and deep residual network, and the functionality of the 
"_shortcut connections_" in the residual building blocks.

### :open_file_folder: Dataset

* Input images are 32x32px (width x height) in RGB format (3 channels), which is a Tensor of shape `torch.Tensor([3, 32, 32])`.

* The dataset consists of 50k training images and 10k test images, classified in 10 classes.

* The data will be augmented padding 4px on each side, followed by a random crop of a window of shape 32x32 either from the 
original image or from its horizontal flip; just for the training data.

### :brain: Architecture

* The architecture is summarized in the following table, where `n=3` leading to a neural network with 20 weighted layers.

  | output map size | 32 x 32 | 16 x 16 | 8 x 8 |
  |-----------------|---------|---------|-------|
  | # layers        | 1 + 2n  | 2n      | 2n    |
  | # filters       | 16      | 32      | 64    |
  
* Both, the plain neural network and the residual neural network, have the exact same architecture.

* The numbers of the convolutional filters are 16, 32, and 64; so that the size of the input image goes 
from 32x32 to 16x16, and then to 8x8.

* The neural network starts off with a convolutional layer which applies a 3x3 convolution, resulting in 
16 out channels.

* Then we will include the residual blocks (the basic, not the bottleneck ones), that is a stack of `6*n`
layers with 3x3 convolutions, that contains `2n` layers for each feature map size.

* The subsampling/downsampling is performed by convolutions with a stride of 2, instead of using pooling operations. 
A comparison between both approaches can be found at: 
[Stackexchange: Pooling vs. stride for downsampling](https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling/387522)

* Finally, the neural network ends with a global average pooling and a fully connected linear layer with 10 units
that stands for the 10 classes of the CIFAR10 dataset.

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

* [Howard, J., Gugger, S., &amp; Chintala, S. (2020). Chapter 14. ResNets. In Deep learning for coders with fastai and PyTorch: AI applications without a PhD (pp. 441–458). O'Reilly Media, Inc.](https://www.amazon.es/Deep-Learning-Coders-Fastai-Pytorch/dp/1492045527)

### :computer: From code

* [Official PyTorch implementation of ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
