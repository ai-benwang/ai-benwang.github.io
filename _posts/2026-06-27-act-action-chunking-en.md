---
title: "ACT: Using Action Chunking to Make Robot Imitation Learning Easier"
date: 2026-06-27
permalink: /posts/2026/06/act-action-chunking-en/
excerpt: "An intuitive explanation of how ACT uses action chunks and temporal ensembling to reduce compounding error in robot imitation learning."
lang: en
canonical_post: true
related: false
categories:
  - Robotics
tags:
  - ACT
  - Imitation Learning
  - Robotics
  - ALOHA
---

<p class="language-switch"><strong>Language:</strong> <a href="/posts/2026/06/act-action-chunking-cn/">中文</a> | English</p>

Traditional fine-grained robotic manipulation requires high-precision closed-loop feedback and strong hand-eye coordination. The robot has to keep adjusting and replanning its actions as the environment changes. An error of only a few millimeters can be enough to make the entire task fail. For this reason, existing fine-manipulation systems often depend on expensive robots and high-end sensors to achieve accurate state estimation. KUKA, for example, describes a robotic assembly process for high-voltage battery pack connectors, where handling flexible components requires extremely precise position and torque control [1].

At the same time, fine-grained manipulation usually involves objects with complex physical properties. In many such cases, learning the manipulation policy directly is much simpler than building a full physical model of the environment [2]. This is why recent robot imitation learning methods have increasingly shifted toward policy learning. Among them, ACT, or Action Chunking with Transformers, is one of the most representative approaches.

Imitation learning, however, faces two major problems [2].

The first problem is compounding error. A very small error in a predicted action sequence can push the robot into a noticeably different state. Once that happens, later predictions are made from states that the model did not really see during training, and the error keeps accumulating.

This is easy to understand: one wrong step can lead to another. As the errors build up, the states encountered during inference drift farther and farther away from the states in the training data, and the model becomes less able to make correct predictions.

The second problem is the presence of temporally correlated confounders in human demonstrations, such as occasional pauses during teleoperation. These behaviors are hard to model with a traditional Markovian policy that predicts only one action at a time.

Here is a simple example. Suppose a person is placing a battery into a battery holder. Once one end of the battery is already in place, the other end still needs to be pressed down. Sometimes the person may pause briefly before pressing it down; sometimes they may not pause at all. If we collect several demonstrations like this and use them to train a model, the model may become confused. Some trajectories contain a pause, while others do not. In order to minimize average error, the model may eventually learn an awkward motion or even produce jitter.

ACT introduces the idea of action chunking from neuroscience. The basic idea is that humans often organize a series of individual movements into a single larger unit, which can then be stored and executed more efficiently. Intuitively, ACT learns the following: given the current observation, how would the human operator control the robot over the next several time steps?

Take the battery placement example again. For a human operator, the task can be split into several action chunks: pick up the battery, place one end into the holder, and press down the other end. If ACT can learn in terms of these human-like chunks, it brings two benefits:

1. It shortens the effective decision horizon, which helps reduce error accumulation.
2. It reduces the impact of non-Markovian behavior on imitation learning.

The authors also propose **Temporal Ensembling** as an optional technique. The model predicts multiple overlapping action chunks and aggregates them to produce a smoother action trajectory.

So in practice, ACT can be understood as the following process. Suppose the model predicts 10 future time steps each time:

1. At time $T_0$, it predicts the next 10 actions.
2. At time $T_1$, it again predicts the next 10 actions.
3. The same process continues at every time $T_i$, where the model predicts a 10-step action chunk starting from the current time.
4. For a given execution time, the action actually sent to the robot is a weighted average of the predictions made at different earlier time steps for that same moment.

![Action Chunking algorithm diagram](/images/blog/act/action_chunking.jpg)

Figure 1: Action Chunking algorithm diagram [2].

## Data Collection and Training Target

In the ACT paper, the authors use ALOHA, a Leader-Follower robot system [2].

During data collection, the recorded action is the joint position of the Leader arm. This is an important detail: the model uses the Leader joint positions, not the Follower joint positions, as the training actions. The reason is that the force applied by the robot to the environment is implicitly determined by the position error between the Leader and the Follower through the low-level PID controller. The Leader joint position therefore better reflects the target motion that the human operator actually intends the robot to execute.

My understanding is that we do not directly learn the Follower joint positions because the Follower may deviate from the Leader due to contact, resistance, or other environmental factors. Learning the Leader angles is essentially learning the operator's intended target position. The gap between the Leader and the Follower can then be converted, implicitly through the low-level PID controller, into an appropriate contact force. Intuitively, the farther the Follower is from the target, the larger the control effort may become.

This can also be understood from the following equation:

$$
\tau = K_p(q_{\text{target}} - q) + K_d(\dot{q}_{\text{target}} - \dot{q}) + K_i\int \text{error}
$$

where:

- $q_{\text{target}}$ is the target joint position output by ACT, which corresponds to the Leader joint position.
- $q$ is the current actual joint position of the Follower.
- $\tau$ is the motor torque.

The most important term is the first one:

$$
K_p(q_{\text{target}} - q)
$$

In other words, the larger the position error, the harder the PID controller pushes. ACT is in fact controlling servos, and the servos already have their own internal position PID control. ACT only tells the servo where it should go. How much force is used to get there is decided automatically by the internal controller of the Dynamixel motors used in the paper, based on the position error. From this perspective, ACT learns how a human operator would control the robot over the next several time steps, given the current observation.

## Model Choice

ACT uses a CVAE, or Conditional Variational Autoencoder, as its training framework. During training, the CVAE encoder learns a latent variable $z$ from the current robot state and the demonstrated action sequence. This latent variable represents different possible styles for completing the same task. The Transformer decoder then combines the current observation, including visual features and robot state, with the latent variable $z$ to predict a future action chunk.

![CVAE training and inference diagram](/images/blog/act/cvae.jpg)

Figure 2: CVAE training and inference diagram.

During inference, the true future action sequence is no longer available, so the CVAE encoder is not used. Instead, the latent variable $z$ is set directly to the mean of the prior distribution, which is the zero vector. The image still goes through a visual backbone such as ResNet to extract features, and those features are fed into the Transformer decoder together with the robot state. The decoder then generates the future action sequence.

A natural question is: why not just use a sequence model to encode and decode actions directly? Why do we need to learn this latent variable $z$ at all?

The key reason is that robot behavior can be inherently multi-modal. A robot may have several valid ways to complete the same goal. For example, to pick up an object or a cup, it could approach from the left or from the right. If we give all of these demonstrations to the model without distinguishing between them, the model may end up learning an average behavior, such as approaching from the middle. This is why we need to learn the latent variable $z$ during training: it captures the style of the action. During deployment, because we usually want deterministic behavior, $z$ is set to the zero vector.

Because a detailed explanation of the CVAE algorithm would require more space, the next chapter will use the ACT implementation in Hugging Face LeRobot to walk through the training and inference process in more detail [3].

## References

[1] KUKA. Liebherr battery pack assembly: high-voltage battery. https://www.kuka.com/en-de/industries/solutions-database/2023/05/liebherr-battery-pack-assembly%2C-high-voltage-battery

[2] Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. arXiv:2304.13705. https://arxiv.org/pdf/2304.13705

[3] Hugging Face. ACT (Action Chunking with Transformers) - LeRobot documentation. https://huggingface.co/docs/lerobot/en/act
