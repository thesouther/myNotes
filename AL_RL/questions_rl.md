# 0 面试-RL相关

## AlphaStar

1. 计算优势函数, GAE与UPGO

   - GAE和UPGO都关⼼心如何将多步以后未来的信息纳⼊入现在的 Advantage估计中. GAE使⽤用Soft的形式通过λ项控制未来信息和现 在信息的平衡(即偏差和⽅方差的平衡). UPGO则使⽤用Hard的形式 直接将未来乐观的step的信息纳⼊入Advantage.

2. 计算状态值V

   - TD($$\lambda$$)

## off-policy重要性采样相关

### 1. 重要性采样推导: 

[4 周博磊RL-3-model_free](./zhou_model_free.md#imsamp)
<a href="./zhou_model_free.md#imsamp">重要性采样部分</a>

### 2. 为什么Q-learning 不用重要性采样?

简单来说, Qlearning没有在**策略分布上**对值函数的期望值进行估计.
他是采样版本的值迭代方法, 使用贝尔曼最优方程, 而不是贝尔曼期望方程.
他是在transition分布上进行的, 而不是在策略分布上进行.

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_05_24_23_49_01.png">
</div>
