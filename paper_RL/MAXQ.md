# 2.1 分层强化学习-MAXQ

paper[The MAXQ Method for Hierarchical Reinforcement Learning](http://axon.cs.byu.edu/Dan/778/papers/Hierarchical%20Reinforcement%20Learning/Dietterich1.pdf)

## 1. 核心思想

分层RL好处：
- 提高探索效率： 
- 加速学习效率： 参数少，并且子任务可以忽略不相关的状态特征
- 在新问题上可以学的更快： 可以重用学到的子任务；

为了达到上述目标，目前有三个方面的探索：
- Dean and Lin (1995), 层次化分解, 主要是为了加速计算最优策略.
- Parr and Russell (1998). 设计了一个程序上的包含可能策略的抽象分层; 他们的方法通过有效地使层次结构扁平化，计算出受这些基本约束的最优策略。我们称这种方法为**分级最优**的，因为它的策略在相应的层是最优的.
- Singh (1992)等, 也是程序设计上的分层. 分层定义每个子任务时都有各自的目标状态或终止条件. 每层的子任务对应于自己的MDP, 目标是得到每个子任务的局部最优.我们称其为 **"递归最优"**.  

本文主要基于MAXQ算法介绍第三种方法.

如何解决分层结构中的信用分配问题？
