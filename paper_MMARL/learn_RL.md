# 1.2 learning to reinforcement learn

paper: [Learning to Reinforcement Learn](https://arxiv.org/pdf/1611.05763v1.pdf)

**核心思想:**提出元强化学习, 解决RL任务迁移问题, 提高数据利用效率.

以前工作证明, RNN可以在**完全监督**条件下支持meta-Learning. ==What emerges is a system that is trained using one RL algorithm, but whose recurrent dynamics implement a second, quite separate RL procedure.This second, learned RL algorithm can differ from the original one in arbitrary ways.== 因为该算法是学习到的, 所以它可以在训练域中更好地利用结构信息

目前RL与人类相比两个最大的问题:
- 需要大量数据进行训练.
- 只使用于特定任务, 而人类可以适应变化的任务.

**关键思想是使用标准的RL方法训练RNN网络, 使其自己生成RL程序. 可以提高适应性和样本效率.**

## 1. Method

### 1.1 RNN实现的Meta-Learning

灵活,data-efficient的学习方法需要好的先验偏差(prior biases). 该偏差可以从两种方法得到:
- 设计到学习系统中, 例如卷积网络;
- 也可以通过学习得到. 例如元学习.

meta-learning标准设置: 学习智能体需要面对多个各不相同的任务, 这些任务可以共享一些潜在的规则集合. Mete-Learning定义为: **智能体在每个新任务中, 平均比之前任务更快地提升自身表现的一种效应.**
在架构层面上, meta learning包括两个层次的学习系统: 
- 底层系统用来适应新任务, 学习较快;
- 上层系统学习较慢, 是跨任务学习, 主要用来调优和改进低层的系统.

==meta learning有很多方法, 本文使用Hochreiter提出的方法, 其使用标准BP在一系列相关联的任务中训练RNN. 其关键是, 在一个任务的每一步,网络接收一个辅助输入, 来指示前一个步骤的目标输出.(例如一个回归任务, 在每一步, 网络把x作为输入, 并期望输出相应的y, 但是网络在前一步同时接收一个公开的target y作为输入.) 在这个场景中, 每个训练的episode使用一个不同的函数来生成数据, 但如果所有的函数都是从同一个参数族中提取的, 那么系统就会逐渐调整到这个一致的结构，在episode中越来越快地收敛到精确的输出.==

上述方法中一个有趣的方面是, 每个新任务的学习基础过程完全源自RNN的动力学意义, 而不是用于调整网络权重的反向传播过程(the process that underlies learning within each new task inheres entirely in the dynamics of the recurrent network, rather than in the backpropagation procedure used to tune that network’s weights).在经过一段初始训练后, 即使权重保持不变, 网络也能提高它在新任务上的表现. 
该方法的第二个重要方面是，RNN的学习过程适合于训练网络的任务族的结构，嵌入的偏差使其在处理来自该任务族的任务时能够有效地学习.(A second important aspect of the approach is that the learning procedure implemented in the recurrent network is fit to the structure that spans the family of tasks on which the network is trained, embedding biases that allow it to learn efficiently when dealing with tasks from that family.)

### 1.2 deep meta RL 

