# 4 周博磊RL-8-IRL

模仿学习不能看作是纯监督学习, 因为还是一个序列决策任务, 之后的状态分布和动作决策相关.

一般都是直接拿人的数据进行训练

**但是: 当进入样本没有访问过的状态时, 会出错, 并且错误会累加**

## 1. 解决1:**DAgger: Dataset Aggregation** 

1. 加入更多的数据. 即让训好的模型在环境中运行, 收集出错的状态, 人类再给他指导数据
   - **让训练数据的分布与策略产生的数据的分布一致**
   - 思想: 从策略收集数据, 让人打标签

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_06_06_17_20_38.png">
</div>

2. 缺点: 第三步需要人类打标签, 成本高
3. 改进: 让3步询问其他速度慢但是准确率高的算法.

## 2 Inverse Reinforcement Learning (IRL)

### 2.1 Guided Cost Learning

1. paper:[Finn, et al, ICML'16. https://arxiv.org/pdf/1603.00448.pdf](https://arxiv.org/pdf/1603.00448.pdf)
2. 
<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_06_06_17_27_11.png">
</div>

### 2.2 Generative Adversarial Imitation Learning(GAIL)

1. paper1 [Ho and Ermon, NIPS'16. https://arxiv.org/pdf/1606.03476.pdf](https://arxiv.org/pdf/1606.03476.pdf)
2. paper2 [Finn, Christiano, et al. A connection between GANs, Inverse RL, and Energy-based Models. https://arxiv.org/pdf/1611.03852.pdf](https://arxiv.org/pdf/1611.03852.pdf)

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_06_06_17_29_37.png">
<img width=100% src="img/2021_06_06_17_32_16.png">
</div>

## 3 IRL 改进

### 3.1 基础改进

改进方向:

1. Multimodal behavior:对于一个状态可能有多个解
   - 可以使用多峰的混合高斯分布
   - Latent variable models 或者 Autoregressive discretization
2. Non-Markovian behavior
   - 建模整个序列, 而不是只看当前状态---使用LSTM网络等

   <div style="text-align: center; width: 80%; margin:auto; ">
   <img width=100% src="img/2021_06_06_17_37_42.png">
   </div>

### 3.2 Learning from demonstration 

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_06_06_17_39_45.png">
</div>
