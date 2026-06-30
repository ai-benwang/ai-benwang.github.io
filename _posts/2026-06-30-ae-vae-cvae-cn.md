---
title: "从 AE 到 VAE 再到 CVAE：理解机器人学习中的潜在变量建模"
excerpt: "用直观方式解释 Autoencoder、Variational Autoencoder 和 Conditional VAE 的核心区别，并为机器人学习中的潜在变量建模做铺垫。"
permalink: /posts/2026/06/ae-vae-cvae-cn/
lang: zh
canonical_post: false
related: false
categories:
  - Machine Learning
tags:
  - Autoencoder
  - VAE
  - CVAE
  - PyTorch
  - Fashion-MNIST
---

<p class="language-switch"><strong>Language:</strong> 中文 | <a href="/posts/2026/06/ae-vae-cvae-en/">English</a></p>

## Auto-Encoder

Auto-Encoder 是一种无监督式学习模型。基于反向传播算法与最优化方法，利用输入数据本身作为监督，来指导神经网络尝试学习一个映射关系，从而得到一个重构输出。对于 Auto-Encoder 而言，最好的状态就是解码器的输出能够完美地或者近似恢复出原来的输入。

![Autoencoder architecture](/images/blog/ae-vae-cvae/autoencoder.svg)

图 1：AE 架构图。

可以看到，编码器的作用是把高维输入编码成低维的隐变量，从而强迫神经网络学习最有信息量的特征。而解码器的作用是把隐藏层的隐变量还原到初始的输入维度。上面说过，AE 的训练目标即解码器的输出能够和编码器的输入越像越好。

那么现在有一个问题：一个能够把输出还原成输入的神经网络，它究竟有什么价值？

一般来说，由于在 AE（Autoencoder）的过程中，最终的隐变量是低维的，它实际上实现了一个信息的压缩。也就是说，我们通过一个编码器网络，将输入的信息进行了压缩。通过解码器网络还原出压缩出来的信息。那么也很容易理解：如果解码器网络还原出来的信息和编码器网络的输入越一致，那么说明在压缩过程中的损耗越小，这也是我们网络训练的目标。

在实际使用过程中，AE（Autoencoder）通常被用来进行异常检测。也就是说，在 AE 网络进行训练的过程中，它使用的是正常数据，因此它的网络特征能够对正常数据进行很好的特征提取和压缩。在推理阶段，如果我们给它一个异常的数据，它将无法或者很难对其进行压缩。这意味着从解码器端输出的信息，与编码器网络输入的信息之间的差异将会比较大，我们从此就可以进行异常检测。

## 变分自编码器（VAE）

VAE 是为了解决 AE 在隐空间里没有结构，不能够有效地进行采样和生成的问题。

![AE latent space example](/images/blog/ae-vae-cvae/latentspace_ae.svg)

图 2：AE 的隐空间示例图。

可以看到 AE 的隐空间是不连续的，这是因为在训练中没有有效地规范隐空间的分布。这造成了我们无法对隐空间进行采样。换句话说，如果我们对隐空间进行一个随机采样，很可能会出现一个没有意义的重建目标。

在 VAE 中，我们引入了 KL 散度的概念，通过引入隐变量的分布和高斯分布的 Loss，并把它添加到系统训练的整体目标中，它能够约束 Encoder 学到的潜在分布接近标准高斯分布。

![VAE architecture](/images/blog/ae-vae-cvae/vae_architecture.svg)

图 3：VAE 架构图。

![VAE latent space example](/images/blog/ae-vae-cvae/latentspace_vae.svg)

图 4：VAE 的隐空间示例图。

VAE 的示例代码部分见：[`Variational-Autoencoder-for-Fashion-MNIST`](https://github.com/ai-benwang/Variational-Autoencoder-for-Fashion-MNIST)。

## 条件变分自编码器（CVAE）

对于 VAE，如果我们在解码的时候，随机采样输入 z，那么我们会发现生成的结果也是随机的，不可控。为了解决这个问题，可以有两个做法：

1. 训练一堆 VAE，每个 VAE 只训练一个类别的样本来进行训练。这样的话，如果想生成哪个类别的样本，我们就使用它对应的 VAE 模型。
2. 在 VAE 模型的输入和采样的时候，加入样本的类别信息。

上述的第二种做法就是所谓的 CVAE。

![CVAE architecture](/images/blog/ae-vae-cvae/cvae.svg)

图 5：CVAE 架构图。

在 CVAE 被训练好后，推理时我们将使用解码器部分，通过输入样本的类别信息，解码器中将输出重构的符合该样本分布的结果。

CVAE 的示例代码部分见：[`Conditional-Variational-Autoencoder-for-Fashion-MNIST`](https://github.com/ai-benwang/Conditional-Variational-Autoencoder-for-Fashion-MNIST)。
