# 3 Distilling Policy Distillation

策略蒸馏有很多不同的方法, 不同的蒸馏方法对性能的影响很大. 本文在理论上分析每种变体的 motivation 和性能.本文分析三种基于具体任务的蒸馏方法.

提出**期望熵正则化蒸馏**, 应用范围更广, 且保证收敛.

## 1. Introduction

策略蒸馏, 让学生网络去拟合教师网络产生的基于状态的一个概率分布.

监督学习中: 模型压缩, 重参数化加速推理, 联合训练多个网络.
强化学习中: 训练无法训练的智能体结构, 加速学习, 构建更强的策略, 多任务学习.

策略蒸馏想法简单,但有很多变体, 比如:

- trajectory 从教师策略采样还是从学生采样,或者混合采样;
- 使用教师和学生分布的 KL 损失,还是整个 trajectory probabilities 的 KL 损失.

本文在理论和实验上对比这些思路. 主要贡献如下:

1. 证明常用的从学生网络采样的 trajectory 不能形成一个梯度向量场, 虽然在简单表格环境中有收敛保证, 但是一旦引入奖励, 就会变得震荡. 同时提供恢复梯度向量场的简单方法.
2. 通过实验对比,说明为什么学生驱动的蒸馏是有益的.
3. 在 AC 架构下, 同时对教师的策略和值函数进行蒸馏.
4. 通过理论分析和实验, 给出蒸馏策略选择的决策树.

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_11_02_21_44_38.png">
</div>

## 3. Policy distillations

首先关于更新准则,

<div style="width: 100%; height:100px; line-height:100px; text-align: center; ">
<div style="float: right; width:15%; height:100%; ">
<p>(1)</p>
</div>
<div style="float: right; width:80%; height:100%; ">
<img height=90% src="img/2021_11_03_14_47_39.png">
</div>
</div>

对于一个 control policy $$q$$ 和 $$\hat{R_t} = \sum_{i=t}^{|\tau|} \hat{r_i} = \sum_{i=t}^{|\tau|} \hat{r}(\pi_\theta, V_{\pi_\theta}, \tau_i, a_i, \tau_{i+1}, a_{i+1}, r_i) $$, 不同的 $$l$$ 和 $$\hat{r}$$定义了不同的蒸馏技术.
其中, $$l$$作为辅助损失, 负责当前 step 的策略调整; $$\hat{r}$$ 可以看作内在奖励和外在奖励的组合, 负责长期策略调整.

本文讨论的是无折扣的 episodic RL 问题, 分析过程可以扩展到带折扣的 case 中.

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_11_03_15_05_32.png">
</div>

### 3.1 Control policy

策略蒸馏建模为监督学习问题. 早期研究使用教师策略采样数据进行更新, $$\mathbb{E}$$
