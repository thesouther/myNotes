# 0 面试-ML 相关

## 1. 基本原理问题

### 1.1 方差偏差

测试误差一般有三个来源, 方差, 偏差平方, 噪声方差:

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_07_13_15_16_20.png">
</div>

1. 偏差是采样得到的训练集训练出的模型输出的平均值与真实期望之间的偏差. 通常来自于模型的错误假设.
2. 方差指由采样得到的训练集训练出的所有模型的输出的方差, 与样本分布有关. 通常方差由于模型复杂度相对于样本来说过高导致的.
3. “偏差-方差分解” 说明，泛化性能是由学习算法的能力、数据的充分性以及学习任务本身的难度所共同决定的。给定学习任务，为了取得好的泛化性能，则需使偏差较小，即能够充分拟合数据，并且使方差较小，即使得数据扰动产生的影响小。
4.

### 1.2 梯度爆炸与消失

1. 爆炸:（1）隐藏层的层数过多；（2）权重的初始化值过大
   - 梯度裁剪
   - 正则化
   - ReLU
   - 用 maxout 代替 sigmoid
   - 预训练+微调
2. 消失，（1）隐藏层的层数过多，（2）采用了不合适的激活函数
   - 正交初始化, 高斯初始化
   - 不用 sigmoid-->> ReLu
   - LSTM 的门控机制
   - ResNet 的残差机制
   - BN 层

### 1.3 过拟合

1. 模型角度
   - Dropout, 正则化, 简化模型, DT 剪枝, BN 层
   - 集成学习
2. 数据增强

### 1.4 神经网络

1. BN 层作用

   - 解决 Covariate shift 问题, 在激活函数之前用. 让网络每一层输出的分布一样, 都是均值为 0, 方差为 1.
   - 加速收敛, 控制过拟合(可少用 dropout), 降低网络初始化的敏感度, 允许较大学习率, 跳出局部最优
   - BN 只考虑相对差异, 分类可用, 但是想图像超分辨率等任务不能用 BN
   - 为什么只用在激活函数之前: 归一化无法消除激活函数方差的变化; 而 FC 或者 CNN 层的输出一般是对称\非稀疏的分布, 更加类似于高斯分布, 归一化可以得到更稳定的分布. (比如 ReLU, 高斯分布经过 ReLu 之后分不会变得很不一样.)

2. LN 层作用

   - 也是加速收敛, 用于 RNN
   - 每个样本内做标准化, 不受 batch_size 影响

3. ReLU

   - 优点: 计算复杂度低, 解决梯度爆炸\消失问题, 有利于稀疏表达
   - 缺点: LR 较大时导致神经元死亡问题-->leaky ReLU
   - RNN 中, W 应该初始化为单位阵.

4. NN 参数初始化方法:
   - 全 0 初始化:
   - 随机初始化: 高斯\均匀分布初始化, 正交初始化
   - Xavier 初始化: 保证前向传播和反向传播时每一层的方差一致; 但是假设一个是激活函数是线性的, 这并不适用于 ReLU, sigmoid 等非线性激活函数; 另一个是激活值关于 0 对称, 这个不适用于 sigmoid 函数和 ReLU 函数它们不是关于 0 对称的.
   - 凯明初始化

### 1.5 稀疏表示与压缩感知

1. 任意一个信号都可以在一个过完备字典上稀疏线性表出
2. 稀疏表示就是将一个复杂的数据简单化，压缩感知是要从一个简单化的数据表示中得到复杂的原始数据
   - 使用少量基本信号的线性组合表示某一目标信号, 称为信号的稀疏表示;
   - 用低维的采样数据向量回复或重构 $$N_{quist}$$ 速率采样的高维数据向量, 称为压缩感知.
3. 稀疏表示有助于**实现特征的自动选择, 即通过参数置零过滤一些不重要的特征, 提高泛化能力**.
4. **梯度截断也可以产生稀疏性.**

### 1.6 L1 与 L2 正则化

==**结构风险最小化: 在经验风险最小化的基础上(也就是训练误差最小化), 尽可能采用简单的模型, 以此提高泛化预测精度.**==

1. L1 正则化
   - 最终加入 L1 范数得到的解一定是某个菱形和某条原函数等高线的切点.
   - 经过观察可以看到, 几乎对于很多原函数等高曲线, 和某个菱形相交的时候及其容易相交在坐标轴(比如上图), 也就是说最终的结果, 解的某些维度及其容易是 0, 比如上图最终解是$$w=(0, x)$$, 这也就是 L1 更容易得到稀疏解的原因;
   - 加上 L1 范数容易得到稀疏解(解向量中 0 比较多)
2. L2 正则化惩罚权重变大的趋势

   - 加上 L2 正则相比于 L1 正则来说, 得到的解比较平滑(不是稀疏), 但是同样能够保证解中接近于 0(但不是等于 0, 所以相对平滑)的维度比较多, 降低模型的复杂度.

   <div style="text-align: center; width: 80%; margin:    auto; ">
   <img width=100% src="img/2021_06_02_21_52_59.png">
   </div>

### 1.7 生成模型与判别模型

1. 判别模型:判别方法由数据直接学习决策函数 $$f(x)$$ 或者条件概率分布 $$P(y|x)$$ 作为预测的模型
   - **线性回归、对数回归、线性判别分析、支持向量机**、 boosting、条件随机场、神经网络
   - 优点: 需要的样本量少; 直接学习条件概率, 可以简化问题; 准确率高
   - 缺点: 黑盒操作; 不能反映训练数据本身的特性; 不适用有隐变量的问题
2. 生成模型: 由 X 和 Y 的联合概率分布, 通过贝叶斯公式求得条件概率: $$P(y|x)=p(x,y)/p(x) = p(x|y)p(y)/p(x) $$
   - 隐马尔科夫模型、**朴素贝叶斯模型、高斯混合模型**、 LDA、 Restricted、Boltzmann Machine.
   - 优点: 收敛速度比较快, 即当样本数量较多时, 生成模型能更快地收敛于真实模型; 能够应付存在隐变量的情况, 比如混合高斯模型.
   - 缺点: 联合分布能够给出更多信息, 但是需要更多计算.

## 2. 特征处理

### 2.1 归一化(normalization)

1. 基本用了正则化, 必须使用归一化
2. normalization 方法
   - min-max normalization
   - z-score normalization
3. 所有需要用梯度下降的模型通常都需要进行归一化，比如线性回归、逻辑回归、支持向量机、神经网络等。但决策树模型并不需要.

### 2.2 评估指标

1. 准确率: 样本不均衡时, 占大多数的类别影响大

   - 样本方面: 重采样, 欠采样
   - 使用每个类别下的准确率
   - 使用其他指标: F1Score, 使用 PR 曲线或者 ROC 曲线

2. 余弦距离.欧式距离体现数值上的绝对差异，而余弦距离体现在方向上的差异。

3. PR 曲线与 ROC 曲线/AUC

   <div style="text-align: center; width: 80%; margin: auto; ">
   <img width=100% src="img/2021_07_13_15_38_01.png">
   </div>

### 2.3 样本不均衡

<div style="text-align: center; width: 80%; margin: auto; ">
<img src="img/2021_06_02_21_00_06.png">
</div>

## 3. 信息论

**KL 散度=交叉熵-熵**

交叉熵、KL 散度都不具备对称性

**KL 散度可以被用于计算代价, 而 KL 散度=交叉熵-熵, 在特定情况下最小化 KL 散度等价于最小化交叉熵. 交叉熵的运算更简单, 所以用交叉熵来当做代价.**

### 3.1 熵

表示事件 X 的自信息量
$$H(x) = -\sum p(x) \log p(x)$$

1. 熵只依赖于随机变量的分布,与随机变量取值无关，所以也可以将 X 的熵记作 H(p)。
2. 令 0log0=0(因为某个取值概率可能为 0)。

### 3.2 条件熵

已知随机变量 X 的条件下随机变量 Y 的不确定性

条件熵 H(Y|X) 相当于联合熵 H(X,Y) 减去单独的熵 H(X)，即 H(Y|X)=H(X,Y)−H(X)

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2021_07_14_20_48_54.png">
</div>

### 3.2 交叉熵

可以用来表示从事件 A 的角度来看, 如何描述事件 B, 适用于衡量不同事件 B 之间的差异

$$H(p, q) = -\sum p(x) \log q(x)$$

对于不同的事件 B, 计算事件 AB 的 KL 散度时都同时减去事件 A 的熵(KL 散度=交叉熵-熵(A)), 因此, 如果只是比较不同 B 事件之间的差异, 计算交叉熵和计算 KL 散度是等价的.

### 3.3 KL 散度(相对熵)

可以用来表示从事件 A 的角度来看, 事件 B 有多大不同, 适用于衡量事件 A, B 之间的差异.

$$D_{KL}(p||q) = \sum p(x)\log p(x) - \sum p(x) \log q(x) = \sum p(x) \log \frac{p(x)}{q(x)}$$

性质:

- KL 散度在 p(x)和 q(x)相同时取到最小值 0, 两个概率分布越相似, 则 KL 散度越小.
- 不对称性
- 非负性

### 3.4 MSE 和 CE 的区别

1. MSE 衡量两个距离的远近, 经常用来做回归任务.
2. CE 作为 KL 散度的近似, 衡量两个分布的相似度, 经常用来做分类任务.
3. 分类任务中一般不使用 MSE, 因为
   - MSE 作为分类的损失函数会有梯度消失的问题.
   - MSE 是非凸的，存在很多局部极小值点。

举例, 以最简单的逻辑回归为例,

<div style="width: 100%; height:100px; line-height:100px; text-align: center; ">
<div style="float: right; width:15%; height:100%; ">
<p>(1)</p>
</div>
<div style="float: right; width:80%; height:100%; ">
<img width=100% src="img/2021_11_01_20_55_44.png">
</div>
</div>

<div style="text-align: center; width: 90%; margin: auto; ">
<div style="background: #4cc; width: 100%; height: 30px; text-align: left; ">
<p style="color:white; margin-left: 10px; "><b>MSE</b></p>
</div>
<div style="width: 100%; background: #ddd;">
<img width=100% src="img/2021_11_01_20_57_56.png">
</div>
</div>

<div style="text-align: center; width: 90%; margin: auto; ">
<div style="background: #4cc; width: 100%; height: 30px; text-align: left; ">
<p style="color:white; margin-left: 10px; "><b>CE</b></p>
</div>
<div style="width: 100%; background: #ddd;">
<img width=100% src="img/2021_11_01_20_58_28.png">
</div>
</div>

<div style="text-align: center; width: 90%; margin: auto; ">
<div style="background: #4cc; width: 100%; height: 30px; text-align: left; ">
<p style="color:white; margin-left: 10px; "><b>关于MSE非凸问题</b></p>
</div>
<div style="width: 100%; background: #ddd;">
<img width=100% src="img/2021_11_01_21_53_42.png">
<img width=100% src="img/2021_11_01_21_53_54.png">
</div>
</div>

## 4. SVM

### 4.1 SVM 与 LDA 的区别和联系

1. LDA 和线性 SVM 都希望能最大化**异类样例间距**，但 LDA 是异类中心间距最大化，而线性 SVM 考虑的是支持向量间距最大。
2. LDA 的目标函数考虑了同类样例的协方差，希望同类样例在投影空间尽可能靠近，而线性 SVM 却没有考虑这一点。
3. 关于数据是否线性可分的问题, 如果使用软间隔的线性 SVM, 线性可分这个条件是不必要的, 如果是硬间隔线性 SVM, 那么线性可分是必要条件. **但是 LDA 不管数据是否线性可分, 都可以进行处理**.
4. 假如当前样本线性可分，且 SVM 与 LDA 求出的结果相互垂直。则当 SVM 的支持向量固定时，再加入新的非支持向量样本，并不会改变 SVM 中求出的 w。但是新加入的样本会改变原类型数据的协方差和均值，从而导致 LDA 求出的结果发生改变。这个时候两者的 w 就不再垂直，但是数据依然是可分的。所以, 线性可分 LDA 求出的 wl 与线性核支持向量机求出的 ws 垂直，这两个条件是不等价的。
