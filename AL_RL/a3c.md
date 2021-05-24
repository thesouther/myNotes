# 1 经典-A3C

paper: [2016-Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf)

* online RL与DNN结合一般不稳定. 
  + 因为online RL训练时产生的序列观察数据是非平稳随机过程; 并且RL算法更新过程具有高度相关性; 
  + 一般用经验池机制解决, 去相关; 
  + Replay-Buffer机制缺点: 每次迭代使用更多内存和计算, 算法必须是off-policy的.
* 并行化对于RL来说稳定性更好. 
  + 训练加速, 只用多核CPU训练, 时间比单GPU减半, 计算资源比分布式计算系统少; 
  + **数据去相关**, 每个时间步不同智能体经历不同的状态; 
  + A3C效果最好, 适用离散\连续动作空间, 适用于前馈网络和循环网络.

## 相关工作

1. Gorila. 

   + 分布式Actor在各自环境副本行动; 
   + 一个离散repaly buffer; 
   + 一个learner从buffer中抽样, 训练DQN loss相对于策略参数的梯度.
   + 梯度异步发送给一个中心的参数server, 更新model的central copy.
   + 然后更新后的策略参数以固定间隔发送给actor-learner.
   + 使用100个离散actor-lerner进程, 30个parameter server, 共130台机器.
   +  Gorila在49个游戏上超过DQN, 很多游戏Gorila比DQN快20倍.

2. 以前的并行化方法.

   + 使用Map-Reduce机制, 加速线性逼近时batch计算速率, 主要加速矩阵计算, 而不是提高数据收集效率或提高稳定性.
   + 并行化版本sarsa. 多个离散actor-lerner加速训练, 每个actor-lerner分别训练, 基于通信周期性地把那些比其他lerner变化巨大的更新发送给权重.
   + 1994年Tsitsiklis, 研究了异步Q-learning的收敛性, 其结果显示, Q-learning在某些信息过时的情况下仍然保证收敛, 只要保证最终将过时信息丢弃.
   + 其他并行化方法, 主要是**进化算法**, 可以用分布式适应度函数直接并行化.
