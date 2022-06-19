# 0. 综述-Graph Learning based Recommender Systems (GLRS)

paper: [Graph Learning Approaches to Recommender Systems: A Review]()

## Introduction

推荐系统中的用户/item/属性之间存在显式或隐式地关系,是一个天然的图结构, 图模型(随机游走或者图神经网络)可以很好地处理关系数据, 图学习主要建模用户偏好和意图/item 特征和受欢迎程度, 可以提升 RS 的准确性/可靠性/可解释性(因果推断能力).

GLRS 形式化: 数据形式根据不同任务和属性有所不同. 因此从更高的视角进行形式化. 设图$$G=\{V,E\}$$, user 和 item 是节点, 关系是边, 推荐系统通过建模改拓扑结构, 形成推荐结果.

$$
R=argmax(G) \qquad \qquad (1)
$$

根据不同的场景和数据, 图 G 可以是同构异构的, 静态/动态的, 推荐结果 R 的形式也多种多样, 可以是 predicted rating 或 ranking. 优化目标也不同: they could be the maximal choice utility according to graph topological structure or the maximal probability to form links between nodes.

## 2 Data Characteristics and Challenges

|                      数据类型                       |                                     实例                                     |                   推荐任务                    |                                     挑战                                     |                                                                                                     备注                                                                                                     |                                               方法                                               |
| :-------------------------------------------------: | :--------------------------------------------------------------------------: | :-------------------------------------------: | :--------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|                  RS on Tree Graphs                  |                           Item hierarchical graphs                           |               Rating prediction               |                               如何学习层次关系                               |                                                                      物品分类级别, 类别/子类/样品..., 可以提升推荐丰富度, 避免重复推荐                                                                       |                                Knowledge graph [Gao et al., 2019]                                |
| RS on Unipartite Graphs (user 或 item 的同构单部图) |          用户社交关系图/物品共现(共同出现在一个订单或 session 里)图          |           朋友推荐, next-item 推荐            |            如何学习用户内在关系和影响传播关系/如何学习物品间关系             |                                                                                                                                                                                                              | Random walk [Bagci and Karagoz, 2016], Graph neural networks [Wu et al., 2019b; Xu et al., 2019] |
|            RS on Bipartite Graphs 二部图            |         user-item 交互图, 同构(只有一种交互关系)或异构(多种交互关系)         |                  Top-N 推荐                   |                                                                              |                                                          建模为通过已知边预测未知边的问题, 还要考虑不同类型关系之间的影响(例如点击对购买行为的影响)                                                          |                                 Random walk [Li and Chen, 2013]                                  |
|               RS on Attributed Graphs               |                         用户属性图谱, item 属性图谱                          |               朋友推荐/社交推荐               |                                                                              |                                                属性图是异构图, 包含两种边, 一种是用户/item 之间的关系, 一种是与属性相连的边, 用户/item 通过共同的属性间接相关                                                |   Graph representation learning [Verma et al., 2019], Graph neural networks [Fan et al., 2019]   |
|         RS on Complex Heterogeneous Graphs          | User-item interaction graphs combined with social relations or item features |        社会推荐, 评价预测, top-n 推荐         |                   通过异构信息将两个图联系起来很具有挑战性                   | 解决 user-item 交互数据的稀疏性, 通常将用户关系(social RS)或物品特征(冷启动)与交互信息结合, 形成两种异构图(user-item 交互二部图, user 社交图/item 特征图), **两个图中共用的物品或用户作为两个图的桥梁节点.** |         Knowledge graph [Palumbo et al., 2017], Graph neural networks [Han et al., 2018]         |
|       RS on multi-source heterogeneous graphs       |        Attributed multiplex heterogeneous graphs (带属性的多重异构图)        | Rating prediction, Top-n item recommendations | 如何整合多种异构图谱数据?如何从多源异构图提取相关信息, 减少噪声和不相干信息? |                                                        user-item 二部图(提供用户偏好和选择信息),用户属性图, 用户社交图谱, 物品属性图谱, 物品共现图谱                                                         |                         Graph representation learning [Cen et al., 2019]                         |

## 3 Graph Learning Approaches to RS

主要分类:

1. Random Walk Approach, 通过在图中随机游走更新节点权重, 边权隐式表现节点的偏好或交互传播过程, 然后基于更新后的概率对节点排序.
   - 优点: 捕获节点之间复杂的高阶/间接的关系
   - 经典算法:
     - 游走方式: basic random walk based RS [Baluja et al., 2008], random walk with restart based RS [Bagci and Karagoz, 2016; Jiang et al., 2018] , 每次转移以固定概率回到开始节点.
     - 转移概率: [Eksombatchai et al., 2018] 每一步都计算一次特定用户转移概率, 提升个性化推荐.
   - 应用:
     - [Gori et al., 2007], item ranking
     - [Nikolakopoulos and Karypis, 2019] 用 user-item 二部图进行 top-n 推荐, 用 item-item 近邻关系建模转移概率
   - 缺点
     - 每个用户单独游走进行 rank, 代价太大, 不适合大型系统
     - 是一种启发式方法, 没有优化目标, 效果不好.
2. 图表示学习, 将节点映射进低维的隐层表示, 来编码结构信息.
   - Graph Factorization Machine based RS (**GFMRS**), 图因子分解机.
     - 思路: 首先基于图的 meta-path 将内部节点的 commuting 矩阵进行分解 ,得到节点的隐式表示, 作为 RS 的输入 [Wang et al., 2019d].
     - 优点: 学习到了(异构)节点间的复杂关系.
     - 缺点: 受数据稀疏性影响
   - Graph Distributed Representation based RS (**GDRRS**), 图分布表示. 基本都 fellow Skip-gram [Mikolov et al., 2013]模型.
     - 代表:
   - Graph Neural Embedding based RS (**GNERS**), 图神经嵌入.
3. GNN
   - Graph Attention network based RS (**GATRS**), 图注意力网络.
   - Gated Graph Neural Network based RS (**GGNNRS**), 门控 GNN.
   - Graph Convolutional Network based RS (**GCNRS**), 图卷积网络.
4. 知识图谱
   - Ontology based KGRS (**OKGRS**), 基于实体的方法.
   - Side information based KGRS (**SKGRS**), 基于边信息的方法.
   - Common knowledge based KGRS (**CKGRS**), 基于公共知识的方法.

|         方法         | 核心思想 | 优点 | 缺点 | 代表 |
| :------------------: | :------: | :--: | ---- | ---- |
| Random Walk Approach |          |      |
|      图表示学习      |          |      |
|         GNN          |          |      |
|       知识图谱       |          |      |
|                      |          |      |
|                      |          |      |

https://blog.csdn.net/abcdefg90876/article/details/105885234/
https://mp.weixin.qq.com/s/9X7TENKPV0MMjIW_Eb9xxg

https://mp.weixin.qq.com/s/tyXx3cnlMjk87HE3XzJ4wg
https://blog.csdn.net/Andre_Jan/article/details/120195685
https://zhuanlan.zhihu.com/p/114798371
