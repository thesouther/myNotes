# 1.1 LIIR

学习个体的固有奖励

论文: [LIIR: Learning Individual Intrinsic Reward in
Multi-Agent Reinforcement Learning](http://papers.nips.cc/paper/8691-liir-learning-individual-intrinsic-reward-in-multi-agent-reinforcement-learning.pdf)

以前方法:
- 设计奖励函数形式的方法;
- 使用集中式Critic解决信用分配问题.

**本文: 内在视角下的奖励. 结合上述两种方法, 通过最大化集中式critic, 让每个智能体学习参数化的个体固有奖励函数.**
- 不是像QMIX等算法那样专注于设计Q值函数. 而是每一步每个智能体都学习自己内在奖励.
- 为了达到这个目的, 每个智能体还要计算一个单独的proxy critic引导策略网络更新.
- 同时, 参数化固有奖励函数, 其输出固有奖励, 朝最大化期望累计团队奖励(集中critic)的方向更新, 其目标函数与原始MARL问题一样.
- 该方法对于值函数没有过多的假设 , 而是使用明确的即时奖励分配信用.

环境:SC2

## 总述

传统MARL是在团队视角下最大化奖励函数. 通过一个中央控制器根据全局状态进行控制, 多智能体主要解决通信方案问题.

**本文焦点**: 当没有通信时, 智能体只根据自己局部观察进行预测, 如何通过全局奖励, 让多智能体动作具有多样性.

**困难**:
- 没有中央控制器, 单个智能体根据局部观察学习合作策略很难;
- 通常MARL任务中只有团队奖励, 信用(贡献)分配困难.

**目标**: 在只有团队奖励的环境中, 把内在奖励函数引入MARL中, 用来区别每个智能体的贡献. 
- 为每个智能体学习一个参数化内在奖励函数, 其每一步输出是内在奖励, 用于让智能体产生多样化的行为;
-  有了内在奖励，为每个agent定义了一个不同的proxy期望折现回报, 它是来环境的真实团队奖励和学习到的内在奖励的组合.
-  使用AC方法, 每个智能体的策略网络在相应proxy critic指导下更新;
-  内在奖励函数的参数使用最大化标准累计折扣团队奖励更新.
-  最终, 整个过程的目标函数与原始的MARL一样.


单个代理目标的解决问题嵌套在最大化标准多代理返回的外部优化任务中


从最优化的视角看, 该模型是二次优化方法. 外层优化任务最大化标准多智能体回报, 内部嵌套优化个体proxy目标. **策略网络的参数和固有奖励网络的参数, 分别作为内部和外部最优化问题的参数.**

研究结果表明，学习的内在奖励函数能够产生不同的奖励信号，agent也能够以协作的方式进行多样化的行为。

## 2. 相关工作

每个智能体应该基于自己的局部观察学习策略, 难点在于解决信用分配问题.
- centralized critic and decentralized policy
- COMA, VDN, QMIX

内在奖励问题, 一些工作对于内在奖励的定义:
- 两个连续状态之差的平方. [Feature control as intrinsic motivation for hierarchical reinforcement learning]()
- 使用好奇心指标作为内在奖励. [Curiosity-driven exploration by self-supervised prediction.]()
- 内在奖励的学习与策略更新相结合. [Optimal rewards for cooperative agents.]()
- 参数化内在奖励函数，并交替更新策略参数和内在奖励参数. [On learning intrinsic rewards for policy gradient methods.]()

## 重点参考:

- 单智能体最优内在奖励问题. 
  - [Intrinsically motivated reinforcement learning: An evolutionary perspective.]()
  - [Reward design via online gradient ascent. ]()
  - [Deep learning for reward design to improve monte carlo tree search in atari games.]()
  - [On learning intrinsic rewards for policy gradient methods.]()