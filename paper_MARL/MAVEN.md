# 5 MAVEN

MAVEN: 多智能体变分探索

paper：[MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/pdf/1910.07483v1.pdf)


同样是基于集中式训练分布式执行的算法

首先了解一些概念，Dec-POMDP, centralised training with decentralised execution(CTDE), [QMIX](./QMIX.md), [QTRAN](./QTRAN.md), [IGM](./QTRAN.md)，这些在之前的内容中讲过。

然后确定一些前提知识：
- 中心化训练时要使用每个智能体的动作-观察历史、以及全局状态；在分布式执行时，每个智能体只是用自己的局部动作-观察历史。
- QMIX使用单调性约束，可以保证在去中心化执行时，只使用$$\mathcal{O}(n|U|)$$的时间
- QTRAN使用线性约束，但是因为约束太松，表现不好。并且计算复杂度很高，在连续任务上状态任务空间中无法求解。
- 对于部分可观察环境来说，IGM限制很严格；但是对于完全可观察话环境，在Q值模型有足够表示能力时，所有的任务都是可分解的。


MAVEN算法思想：关注**在去中心化MARL中不充分的探索引起的问题**。
- 使得单个智能体样本效率降低，去中心化必要的表示性约束，是算法趋于次优。单智能体可以通过增大$$\epsilon-greedy$$探索或策略方差来促进探索；但是在MARL中这些方法不可以。
- committed探索可以用来解决上述问题。在committed exploration中，探索行为通过沿时间步扩展以协作方式执行。在MARL包括长期合作，需要探索行为发现最大化奖励的暂时性联合策略。现有CTDE算法没有考虑committed exploration。
- MAVEN：通过引入分层控制隐空间，混合基于值与基于策略的方法。智能体基于值方法，通过共享隐变量（由分层策略控制）限制其行为。因此固定隐变量，每个联合动作值函数可以看作整个episode里的一个联合探索行为模式。
- 然后，MAVEN最大化trajectories和隐变量之间的互信息，学习上述探索行为的多种集合。
- 这使得MAVEN可以在遵循表示性限制的同时，实现committed exploration。