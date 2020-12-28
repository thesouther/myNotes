# 3.5 游戏AI-绝悟1-NIPS

paper: [Mastering Complex Control in MOBA Games with Deep Reinforcement Learning](https://arxiv.org/abs/1912.09729)

核心: MOBA游戏的动作空间和状态空间都巨大, 绝悟从系统和算法层面进行解决. 
- 系统层面: 通过低耦合性和高可扩展性架构进行高效探索.
- 算法层面: 结合控制依赖解耦,动作掩码, 目标attention, dual-clip PPO等算法(control dependency decoupling, action mask, target attention, and dualclip PPO,), 使用AC架构进行高效训练. 

## 1. 前言

### 1.1 任务复杂度

MOBA 1v1是RTS(real-time strategy)游戏, 复杂度比围棋高几个量级, 
- 复杂性不止来自于动作空间和状态空间, 
- 在策略上还要学会规划,攻击,防御,技能控制, 欺骗对手等. 
- 另外还有野怪防御塔等实体, 让选择目标变得很困难. 
- 另外游戏里有很多不同的英雄, 算法框架必须足够健壮可以适应多种英雄. 
- 最后可用的1v1监督数据也很少.

|  ![](img/2020_12_28_22_10_06.png)  |
| :--------------------------------: |
| Table 1: Comparing Go and MOBA 1v1 |

### 1.2 算法框架

网络整体解读:
- 使用多模态输入, 解耦控制动作相互关系, 探索剪枝机制, 攻击attention机制.
- 在系统层面, 可扩展性的off-policy训练架构.
- 算法层面使用AC架构建模MOBA控制动作.
- 多网络采用multi-label PPO的目标函数进行优化，加上控制依赖解耦, 目标选择的attention机制, 提高探索的动作掩码(action mask)机制, 用LSTM学习技能组合, 改进版dual-clip PPO算法保证收敛性.

本文使用2个智能体的多智能体MDP设定(two-agent world for multi-agent Markov games). 本文使用竞争环境设定. 

目标函数也是$$\mathbb{E}[\sum_{t=0}^T \gamma^tr(s_t,a_t)] $$

## 2 系统设计

首先, 复杂游戏中使用随机梯度方差很大,  要使用大batch的数据加速训练. 所以本文使用松耦合,可扩展性架构并行化使用数据. 系统分为四部分: RL lerner, AI server, 调度模块(Dispatch Module), 记忆池(Memory Pool).

|  ![](img/2020_12_28_23_09_32.png)   |
| :---------------------------------: |
| fig 1 Overview of our System Design |

- AI Server. 算法与环境交互模块, 通过镜像策略使用self-play生成episodes. 对手策略抽样类似于[Emergent complexity via multi-agent competition](). 智能体基于游戏状态特征抽取, 使用Boltzman exploration(即基于softmax分布的抽样)预测英雄动作. 采样后动作被执行,之后游戏核心返回相应的奖励值和下一个状态. **一个AI server使用一个CPU核, 为节省IO成本, 游戏模型推断也在CPU上执行**.为了高效生成episodes,还建立了CPU版的快速推断库FeatherCNN.
- Dispatch Module. 每个调度模块绑定同一个机器的多个AI servers. 从AIserver收集数据样本(reward, feature, action probabilities). 将其压缩打包发送给记忆池.
- Memory Pool. 基于内存的高效循环队列. 支持不同长度的样本，和基于生成时间的数据采样.
- RL lerner, 分布式训练环境. 多个Lerner并行从相同数量的记忆池获得大批量数据, 通过ring allreduce算法对RL学习器中的梯度进行平均. 为了降低IO成本, lerner与pool共享内存, 而不是使用socket通信(提速2-3倍). RL训练模型以peer-to-peer方式快速同步到AI服务器.



## 相关的论文

通过自训练产生合适对手, 然后通过对手训练出负责控制任务的智能体. [Emergent complexity via multi-agent competition]()
通用优势函数. [High-dimensional continuous control using generalized advantage estimation]()
使用历史数据进行模仿学习. [Exponentially weighted imitation learning for batched historical data]()
基于监督学习的MOBA 5v5宏观策略模型. [Hierarchical macro strategy model for moba game ai.]()
基于MCTS的MOBA游戏. [Feedback-based tree search for reinforcement learning]()