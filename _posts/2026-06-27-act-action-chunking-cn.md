---
title: "ACT：用动作分块降低机器人模仿学习的难度"
date: 2026-06-27
permalink: /posts/2026/06/act-action-chunking-cn/
excerpt: "用直观的方式解释 ACT 如何通过动作分块和时间集成降低机器人模仿学习中的误差累积。"
lang: zh
canonical_post: false
related: false
categories:
  - Robotics
tags:
  - ACT
  - Imitation Learning
  - Robotics
  - ALOHA
---

<p class="language-switch"><strong>Language:</strong> 中文 | <a href="/posts/2026/06/act-action-chunking-en/">English</a></p>

传统的机器人精细操作任务需要高精度的闭环反馈，并要求高度的手眼协调能力，以便根据环境变化不断调整和重新规划动作。仅仅几毫米的误差，就可能导致整个任务失败。因此，现有的精细操作系统通常依赖昂贵的机器人和高端传感器，以实现精确的状态估计。KUKA 在其网站上介绍了使用机器人进行高压电池包接插件组装的案例，其中柔性部件的装配需要极高的位置和力矩控制精度 [1]。

与此同时，精细操作通常涉及具有复杂物理属性的物体。在这种情况下，直接学习操作策略，往往比建立整个环境的物理模型更加简单 [2]。因此，近年来的机器人模仿学习算法逐渐转向操作策略学习，其中 ACT（Action Chunking with Transformers）是最具代表性的算法之一。

然而，模仿学习中存在两个主要问题 [2]。

第一，预测动作序列中的一个很小的动作误差，都可能导致机器人状态发生较大的偏离，从而进一步加剧模仿学习中的误差累积（compounding error）问题。

这个问题很好理解，也就是我们经常说的“一步错，步步错”。误差的累积会使推理时的数据状态逐渐偏离训练时的数据状态，从而使模型无法正确地继续推理。

第二，人类示教中存在时间相关干扰（temporally correlated confounders），例如示教过程中偶尔出现的停顿。这类现象很难用传统的、只预测单步动作的马尔可夫策略进行建模。

这里举一个例子：如果我们将一节电池放进电池仓，当电池的一端已经放入、需要把另一端压下去时，有时我们会停顿一下，有时又不会停顿。如果我们采集了若干组这样的示教数据并用于训练模型，模型就会很困惑。因为它看到有些数据中包含停顿片段，而有些数据中又没有。为了使平均误差最小，模型最后很可能会学出某种奇怪的动作或抖动。

ACT 引入了神经科学中的动作分块（Action Chunking）概念。该理论认为，人类会将一系列独立动作组织成一个整体进行存储和执行，从而提高动作执行效率。因此，从直观上理解，ACT 学习的是：在当前观测条件下，人类操作者在未来几个时间步中将如何控制机器人。

仍以“把电池放入电池仓”为例。对于人类操作者而言，这个任务可以分为拿起电池、将电池的一端放入电池仓、压下另一端这几个动作块。对于 ACT 而言，如果能够按照人类的动作块来学习，那么会带来以下两个好处：

1. 降低有效决策长度，从而减少误差累积。
2. 减少非马尔可夫（non-Markovian）行为对模仿学习的影响。

此外，作者还提出了 **Temporal Ensembling（时间集成）** 作为一种可选技术。它会预测多个相互重叠的动作块，并将它们聚合起来，以生成更加平滑的动作轨迹。

因此，对于 ACT 算法而言，真实的执行过程可以理解为如下形式。这里假设模型每次预测 10 个时间步：

1. 在时间 $T_0$，预测之后的 10 步动作。
2. 在时间 $T_1$，同样预测之后的 10 步动作。
3. 以此类推，在每个时间 $T_i$，预测从当前时间开始的 10 步动作。
4. 对于给定的执行时间点，机器人实际执行的动作，是不同预测时刻对该时间点动作预测结果的加权平均。

![Action Chunking 算法示意图](/images/blog/act/action_chunking.jpg)

图 1：Action Chunking 算法示意图 [2]。

## 数据采集和训练目标

在 ACT 论文中，作者采用了 ALOHA 的 Leader-Follower 机器人系统 [2]。

数据采集记录的是 Leader 机械臂的关节位置，并将其作为训练时的动作（action）。这里特别需要注意的是，模型使用的是 Leader 的关节位置，而不是 Follower 的关节位置。这是因为，机器人施加到环境中的力，实际上由 Leader 与 Follower 之间的位置误差通过底层 PID 控制器隐式决定。因此，Leader 的关节位置更能反映操作者真正希望机器人执行的目标动作。

我的理解是：之所以不直接学习 Follower 的关节位置，是因为 Follower 可能由于接触、阻力或其他环境因素，与 Leader 的关节角度存在误差。学习 Leader 的角度，相当于学习操作者期望机器人到达的目标位置；而 Leader 与 Follower 之间还差多少，则可以通过底层 PID 控制器隐式地推导出合适的接触力。直观地说，离目标越远，控制器施加的力可能越大。

也可以从下面的公式来理解：

$$
\tau = K_p(q_{\text{target}} - q) + K_d(\dot{q}_{\text{target}} - \dot{q}) + K_i\int \text{error}
$$

其中：

- $q_{\text{target}}$：ACT 输出的目标关节位置，也就是 Leader 的关节位置。
- $q$：Follower 当前的实际关节位置。
- $\tau$：电机输出的力矩（torque）。

其中最主要的是第一项：

$$
K_p(q_{\text{target}} - q)
$$

也就是说，位置误差越大，PID 控制器越“使劲”。此外，ACT 实际控制的是舵机，而舵机内部自带位置 PID。ACT 只负责告诉舵机“应该去哪里”；至于“用多大劲去”，则由论文中使用的 Dynamixel 电机内部控制器根据位置误差自动决定。从直观上理解，ACT 学习的是：在当前观测条件下，人类操作者在未来几个时间步中将如何控制机器人。

## 模型选择

ACT 采用 CVAE（Conditional Variational Autoencoder，条件变分自编码器）作为训练框架。在训练阶段，CVAE 的 Encoder 根据当前机器人状态和示教动作序列学习潜在变量 $z$，用于表示同一任务中可能存在的不同操作风格；Transformer Decoder 则结合当前观测（包括图像特征和机器人状态）以及潜在变量 $z$，预测未来的一个动作块（Action Chunk）。

![CVAE 训练与推理结构示意图](/images/blog/act/cvae.jpg)

图 2：CVAE 训练与推理结构示意图。

在推理阶段，由于无法获得真实动作序列，CVAE 的 Encoder 不再参与计算，而是直接将潜在变量 $z$ 设置为先验分布的均值，即零向量。随后，图像仍然会经过视觉 Backbone（如 ResNet）提取特征，并与机器人状态一起输入 Transformer Decoder，由其生成未来的一段动作序列。

这里一个自然的问题是：我们为什么不能直接用一个序列模型来编码动作并解码动作输出？为什么一定要学习潜在变量 $z$？

最重要的原因是，机器人的动作本身存在多样化的可能性。也就是说，一个机器人为了完成某个目标，可能会有多种动作路径。比如，为了举起一个设备或水杯，它可以从左边接近，也可以从右边接近。如果不加区分地把这些动作都交给模型学习，那么模型最终可能会学出一个平均值，也就是尝试从中间接近目标。因此，我们必须在模型训练过程中学习潜在变量 $z$，用它表示动作的风格变量。在最后的模型应用过程中，因为我们需要动作具有确定性，所以将 $z$ 置为零向量。

由于 CVAE 的算法细节较为复杂，我们将在下一章结合 Hugging Face LeRobot 中的 ACT 实现，进一步解读该算法的具体训练与推理过程 [3]。

## 参考文献

[1] KUKA. Liebherr battery pack assembly: high-voltage battery. https://www.kuka.com/en-de/industries/solutions-database/2023/05/liebherr-battery-pack-assembly%2C-high-voltage-battery

[2] Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. arXiv:2304.13705. https://arxiv.org/pdf/2304.13705

[3] Hugging Face. ACT (Action Chunking with Transformers) - LeRobot documentation. https://huggingface.co/docs/lerobot/en/act
