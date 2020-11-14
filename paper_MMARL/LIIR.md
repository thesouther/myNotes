# 1.1 LIIR

学习个体的固有奖励

论文: [LIIR: Learning Individual Intrinsic Reward in
Multi-Agent Reinforcement Learning](http://papers.nips.cc/paper/8691-liir-learning-individual-intrinsic-reward-in-multi-agent-reinforcement-learning.pdf)

## 总述

传统MARL是在团队视角下最大化奖励函数. 通过一个中央控制器根据全局状态进行控制, 多智能体主要解决通信方案问题.

本文焦点: 当没有通信时, 智能体只根据自己局部观察进行预测, 如何通过全局奖励,让多智能体动作具有多样性.

以前方法:
- 设计奖励共享方法;
- 使用集中式Critic解决信用分配问题.

本文: 内在视角下的奖励. 结合上述两种方法, 让每个智能体学习个体固有奖励函数. 
- 不是像QMIX等算法那样专注于设计值函数. 而是每一步每个智能体都学习自己固有奖励.
- 为了达到这个目的, 每个智能体还要计算一个单独的proxy critic引导策略网络更新.
- 同时, 参数化固有奖励函数, 其输出固有奖励, 朝最大化期望累计团队奖励(集中critic)的方向更新, 其目标函数与原始MARL问题一样.
- 该方法对于值函数没有过多的假设 , 而是使用明确的即时奖励分配信用.

环境:SC2

从最优化的视角看, 该模型是二次优化方法. 外部优化最大化标准多智能体回报, 内部嵌套优化个体proxy目标. 个体策略和固有奖励函数的参数化, 作为内部和外部最优化问题的参数.