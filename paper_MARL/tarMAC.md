# 2.4 交流-tarMAC

TarMAC: 有目标的多智能体交流

论文: [TarMAC: Targeted Multi-Agent Communication](https://arxiv.org/pdf/1810.11187.pdf)

中心思想:
- 解决发送什么消息和发送给谁的问题.该目标行为通过下游特定任务奖励单独学习,而不用交流监督.
- 采用多轮通信的方法,智能体在行动前进行多轮通信.
- agent学习的目标沟通策略是可解释的和直观的.

## 总体介绍

### Actor(train and test)

总体而言,本文中的每个 agent 每轮不仅要选择一个action还要选一个message.这个message 类似于公网一样被broadcast .

具体到单个的agent,policy网络选用了GRU,输入是该agent的observation和全局的message.输出是根据当前的h给出下一step的动作概率和下一回合传输的message.

### Centralized Critic(train)

Critic每回合拿到GRU的state和各个agent的action,并给出Q.关于为什么不用individual critic,**文章中说是variance,我感觉其实就是会涉及credit assignment.**

**Message**:包含两部分,一个signature和一个value.signature有点像一个公钥,和specific的recipient的私钥相乘后应该会给出比较大的值,而与其他agent的私钥相乘则会很小,用于做value的权重.

此外还提出了multi-stage的通信,没有详细说明,**类似tcp**.

## 1. Introduction

定向交流可能比广播方法有效.使智能体在复杂环境中使用更灵活的沟通策略.

该方法通过一个简单的基于**签名的软注意机制**来操作的: 发送方广播消息和一个密钥,该密钥用来指示要发送给的智能体的属性. 接收方使用该密钥衡量消息的相关性. 这种交流机制使用任务奖励进行端到端训练,在没有attention监督的情况下,隐式学习.

该框架中软注意力机制提供的归纳偏差, 保证智能体能够:
- 交流智能体特定任务目标
- 自适应队伍规模
- 可通过预测的注意力概率解释, 哪个智能体对谁交流什么.

只使用定向通信效果不好,在一个时间步进行多轮通信,涉及到在内存中持久存储,并通过高带宽信道交换大量信息. 所以使用集中式训练分布式执行的AC框架.

本文工作:
- 我们首先对TarMAC及其无attention消融进行基准测试,使用协同导航任务. 表明智能体可以学习跨任务困难的直观注意力行为.
- 在5.2节,在traffic junction环境评估TarMAC. 表明,在团队规模变化的情况下,agent能够自适应地关注"active" agents.
- 5.3节, 在House3D中通过协作的第一人称目标导航任务展示其在3D环境中的效果.
- 在5.4节表明, TarMAC可以与IC3Net结合使用,从而适用混合竞争环境. 并显着改善了性能和抽样复杂性.



## 2. Related Work



## 评价


这一篇的attention本质上是在用本轮自己的observation和之前轮所有agents的observation做的.这不仅仅是没有马尔科夫性的问题了,这个算法根本没有显式地考虑其他agent当前轮的observation,而笔者所学过的绝大多数RL算法都是基于当前轮observation的.

当然了,可以argue上一轮的action和下一轮的observation是强相关的,所以相当于间接考虑了当前轮,这么说有点牵强.