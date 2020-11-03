# 1.1 LIIR

学习个体的固有奖励

论文: [LIIR: Learning Individual Intrinsic Reward in
Multi-Agent Reinforcement Learning](http://papers.nips.cc/paper/8691-liir-learning-individual-intrinsic-reward-in-multi-agent-reinforcement-learning.pdf)

焦点: 如何通过全局奖励,让多智能体动作具有多样性.

以前方法:
- 设计奖励共享方法;
- 使用集中式Critic解决信用分配问题.

本文: 内在视角下的奖励.
结合上述两种方法, 让每个智能体学习个体固有奖励函数.为了达到这个目的, 每个智能体还要计算一个单独的proxy critic引导策略网络更新.同时, 参数化固有奖励函数朝最大化期望累计团队奖励的方向更新, 其目标函数与原始MARL问题一样.

环境:SC2

## 总述

