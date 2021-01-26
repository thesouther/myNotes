# 2 spinup-PPO

核心: 如何提高数据利用率, 最大化策略提升步骤, 同时保证性能函数不崩溃. TRPO使用二阶方法, PPO使用一阶方法保持新旧策略保持一致.

**核心算法性质:**

* PPO是on-policy算法; 
* 适用于离散和连续动作空间环境; 
* 支持MPI实现并行化．

## 速览

### 算法版本

* PPO-Penalty: 把KL散度作为目标函数的惩罚项, (TRPO使用带约束的优化问题), 并在训练过程中自动调整惩罚系数.
* PPO-Clip: 没有对KL散度的惩罚, 而是直接对目标函数进行clip, 让新旧策略不会偏离太远.

### 核心公式

参数更新过程为, 

$$
\theta_{k+1} = \arg \max_{\theta} \mathrm{E}_{s, a \sim \pi_{\theta_k}} \left[ L(s, a, \theta_k, \theta)\right]
$$

使用SGD更新的目标函数为

$$
L(s, a, \theta_k, \theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s, a), \; \; 
\text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s, a)
\right), 
$$

其中$$\epsilon$$表示新旧策略最多离多远的超参数. 上式有一个简单版本:

$$
L(s, a, \theta_k, \theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s, a), \; \; 
g(\epsilon, A^{\pi_{\theta_k}}(s, a))
\right), 
$$

其中

$$
g(\epsilon, A) = \left\{
\begin{array}{ll}
(1 + \epsilon) A & A \geq 0 \\
(1 - \epsilon) A & A < 0.
\end{array}
\right.
$$

### 伪代码

PPO是一种on-policy算法, 也就是说算法根据最新版本的策略版本进行动作抽样. 动作随机性来自初始化和训练过程, 但是随着训练更新, 动作随机性会降低, 容易陷入局部最优.

|<img src="img/2021_01_25_15_22_53.png">|
|:-:|
|fig 1. 伪代码 |
