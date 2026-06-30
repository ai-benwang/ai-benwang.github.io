---
title: "From AE to VAE to CVAE: Understanding Latent Variable Modeling in Robot Learning"
excerpt: "An intuitive explanation of the key differences between Autoencoder, Variational Autoencoder, and Conditional VAE, as a foundation for latent variable modeling in robot learning."
permalink: /posts/2026/06/ae-vae-cvae-en/
lang: en
canonical_post: true
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

<p class="language-switch"><strong>Language:</strong> <a href="/posts/2026/06/ae-vae-cvae-cn/">中文</a> | English</p>

## Auto-Encoder

Auto-Encoder is an unsupervised learning model. Based on backpropagation and optimization methods, it uses the input data itself as supervision to guide a neural network to learn a mapping relationship, and finally obtain a reconstructed output. For Auto-Encoder, the best state is that the decoder output can perfectly or approximately recover the original input.

![Autoencoder architecture](/images/blog/ae-vae-cvae/autoencoder.svg)

Figure 1: AE architecture.

As we can see, the role of the encoder is to encode a high-dimensional input into a low-dimensional latent variable, so as to force the neural network to learn the most informative features. The role of the decoder is to restore the latent variable in the hidden layer back to the original input dimension. As mentioned above, the training goal of AE is that the output of the decoder should be as similar as possible to the input of the encoder.

Now there is a question: what is the value of a neural network that can restore the output back into the input?

Generally speaking, because in the process of AE (Autoencoder), the final latent variable is low-dimensional, it actually implements a kind of information compression. In other words, through an encoder network, we compress the input information. Then through a decoder network, we restore the compressed information. So it is also easy to understand: if the information restored by the decoder network is more consistent with the input of the encoder network, it means that the loss in the compression process is smaller. This is also the goal of our network training.

In practical use, AE (Autoencoder) is usually used for anomaly detection. That is to say, during the training process of the AE network, it uses normal data, so its network features can extract and compress normal data very well. During inference, if we give it an abnormal data sample, it will be unable or find it difficult to compress it. This means that the difference between the information output from the decoder side and the information input to the encoder network will become relatively large, and from this we can perform anomaly detection.

## Variational Autoencoder (VAE)

VAE is designed to solve the problem that AE has no structure in the latent space and therefore cannot effectively perform sampling and generation.

![AE latent space example](/images/blog/ae-vae-cvae/latentspace_ae.svg)

Figure 2: Example of AE latent space.

We can see that the latent space of AE is discontinuous. This is because the distribution of the latent space is not effectively regularized during training. This causes us to be unable to sample from the latent space. In other words, if we randomly sample the latent space, it is very likely to produce a meaningless reconstruction target.

In VAE, we introduce the concept of KL divergence. By introducing the Loss between the latent-variable distribution and the Gaussian distribution, and adding it to the overall objective of system training, it can constrain the latent distribution learned by the Encoder to be close to the standard Gaussian distribution.

![VAE architecture](/images/blog/ae-vae-cvae/vae_architecture.svg)

Figure 3: VAE architecture.

![VAE latent space example](/images/blog/ae-vae-cvae/latentspace_vae.svg)

Figure 4: Example of VAE latent space.

The example code for VAE is here: [`Variational-Autoencoder-for-Fashion-MNIST`](https://github.com/ai-benwang/Variational-Autoencoder-for-Fashion-MNIST).

## Conditional Variational Autoencoder (CVAE)

For VAE, if we randomly sample the input z during decoding, then we will find that the generated result is also random and uncontrollable. To solve this problem, there can be two approaches:

1. Train a group of VAEs, where each VAE only trains on samples from one category. In this way, if we want to generate samples of a certain category, we use the corresponding VAE model.
2. Add the category information of the sample during the input and sampling process of the VAE model.

The second approach above is the so-called CVAE.

![CVAE architecture](/images/blog/ae-vae-cvae/cvae.svg)

Figure 5: CVAE architecture.

After the CVAE is trained, during inference we will use the decoder part. By inputting the category information of the sample, the decoder will output a reconstructed result that conforms to the distribution of that sample.

The example code for CVAE is here: [`Conditional-Variational-Autoencoder-for-Fashion-MNIST`](https://github.com/ai-benwang/Conditional-Variational-Autoencoder-for-Fashion-MNIST).
