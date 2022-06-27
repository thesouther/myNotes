# 8-拉格朗日对偶理论

## 1. "松弛"与"界"

我们面对数学规划问题中的约束项时, 有时可以很直观地感受到一些约束比较简单, 而另一些约束是比较复杂的.

$$
\begin{array}{ll}
\max  & c^{\top} x \\
\text{s.t. }  & A x \leq b  \quad \text {(nice constraints)} \\
& D x \leq d \quad  \text{(complicated constraints)} \\
& x \in X
\end{array}
$$

一个自然而然的想法是能否想办法把这些比较难的约束去掉. 由此引申出优化求解方法中一种重要的思想--"松弛".

关于"松弛", 最容易理解的例子, 莫过于求解线性整数规划问题时, 可以将整数约束转化为整个集合区间上的连续约束, 即"**连续松弛**". 在得到这样一个线性规划松弛问题之后, 可以使用单纯形或内点法等进行快速求解. 之后使用"分支定(上/下)界"的方法, 使用类似于树搜索的方法求解原问题的最优解.

此外, 还可以将困难的约束直接放进目标函数里, 通过引入一个 **price**, 对违反该约束的函数取值增加一些代价. 这就是"**拉格朗日松弛**"的基本思想.

**拉格朗日松弛可以比连续松弛提供更紧的界**

### 1.1 示例 1

我们首先看这样一个问题,

$$
\begin{array}{ll}
\min _{x, y} &c^{T} x+d^{T} y \\
\text{s.t. } &A_{1} x=b_{1} \quad (A1)\\
&A_{2} y=b_{2} \quad (A2) \\
&A_{3} x+A_{4} y=b_{3} \quad (A3)\\
& x,y \in D.
\end{array}
$$

很明显, 约束 A1 和 A2 分别只和 x/y 有关, 而约束 A3 把两类变量耦合在了一起。我们看看如果将 A3 放入目标函数, 问题会转为什么样子:

$$
\begin{array}{ll}
\min _{x, y} &c^{T} x+d^{T} y+\lambda^{T}\left(A_{3} x+A_{4} y-b_{3}\right) \\
\text{s.t.}  &A_{1} x=b_{1} \\
&A_{2} y=b_{2} \\
& x,y \in D.
\end{array}
$$

这其实相当于分别求如下两个优化问题, 再进行求和:

- 子问题 1

$$
\begin{array}{ll}
\min _{x} & c^{T} x+\lambda^{T} A_{3} x \\
\text{s.t.}  & A_{1} x=b_{1} \\
\end{array}
$$

- 子问题 2

$$
\begin{array}{ll}
\min _{y} & d^{T} y+\lambda^{T} A_{4} y \\
\text{s.t.}  & A_{2} y=b_{2} \\
\end{array}
$$

问题看起来简单了很多. 引入的新变量就是上边所说的 price, 即**拉格朗日乘子**, 也称"**对偶变量**"

### 1.2 示例 2 整数规划问题

传统整数规划求解方法包括割平面法/动态规划/分支定界等

但是再大规模整数规划问题下, 这些方法无法在可接受的时间内求解. 因此一般寻求对原问题进行分解或松弛, 之后再使用割平面或者分支定界, 快速求解小规模问题的方法. 主要方法分为如下三类.

1. Benders decomposition (主要思想是行生成+割平面方法)
2. Dantzig-Wolfe decomposition (主要思想其实就是列生成)
3. Lagrangian decomposition (主要思想是 Lagrangian relaxation)

考虑如下 0-1 整数规划问题, 我们看一下使用了拉格朗日松弛后会发生什么:

$$
\begin{array}{lll}
(0-1 \mathrm{IP}) & \min & c^T x,\\
& \text{s.t. } & Ax\le b,  \\
&& x\in \{0,1\}^n \\
\end{array}
$$

将$$Ax \le b$$进行松弛, 引入拉格朗日乘子$$u\in R^m_+$$. 为什么此处的$$u$$取正值, 下一节会进行介绍.

$$
\begin{aligned}
z(u) &=\min \left\{c^{\mathrm{T}} x+u^{\mathrm{T}}(A x-b) \mid x \in\{0,1\}^{n}\right\} \\
&=-u^{\mathrm{T}} b+\min \left\{\left(c+A^{\mathrm{T}} u\right)^{\mathrm{T}} x \mid x \in\{0,1\}^{n}\right\} \\
&=-u^{\mathrm{T}} b+\sum_{i=1}^{n} \min \left\{\left(c+A^{\mathrm{T}} u\right)_{i} x_{i} \mid x_{i} \in\{0,1\}\right\} \\
&=-u^{\mathrm{T}} b+\sum_{i=1}^{n} \min \left\{\left(c+A^{\mathrm{T}} u\right)_{i}, 0\right\}
\end{aligned}
$$

可以看出, 在求解$$z(u)$$的时候, 只需要遍历一遍向量$$(c+A^{\mathrm{T}} u)$$的各个分量, 如果大于 0, $$x$$就取 0; 如果小于 0, $$x$$就取 1. 可以在$$O(n)$$时间内求解.

### 1.3 一般形式的拉格朗日松弛方法

为了更明确地说明拉格朗日乘子在不同情况下的取值, 下面对等式约束和不等式约束进行分别说明, 混合约束的情况下就是把这两种进行组合.

注:

- 我们将**原来的问题称为原始(Primal)问题, 用记号 P 表示**

#### 1.3.1 等式约束线性规划

$$
\begin{array}{llc}
(\mathrm{P}) & \min & f(x) = c^T x,\\
& \text{s.t. } & Ax = b,  \\
&& x  \ge 0 \\
\end{array}
$$

由等式约束$$b - Ax = 0$$, 引入拉格朗日乘子$$\lambda \in R^m $$, 得到

$$
\begin{array}{lc}
\min & g(x) = c^T x + \lambda^T (b- Ax),\\
\text{s.t. } & x\ge 0 \\
\end{array}
$$

此时我们得到了一个在很简单的约束上的目标函数. 将其进行进一步整理, 并定义函数:

$$
\begin{array}{ll}
d&= \min_{x\ge0} \{ b^T \lambda + (c-A^T \lambda)^T x \} \quad (1.3.1) \\
&=  b^T \lambda +  \min_{x\ge0} \{(c-A^T \lambda)^T x \}  \quad (1.3.2)\\
\end{array}
$$

考虑式 1.3.2 右边求极小的项:

$$
\begin{array}{llll}
\text{if} & c-A^T \lambda\ge 0, & d= b^T \lambda, & x^* = 0 \\
\text{if} & c-A^T \lambda < 0, & d=-\infty, & x^* = \infty (\text{infty})\\
\end{array}
$$

显然原始问题转化为一个关于$$\lambda$$的函数, 称为**对偶函数**, 记为$$d(\lambda)$$.

另外一个显然的结论是, 引入松弛项后, 定义域变大了, 实际上是引入了一个原始问题的下界函数$$d(\lambda)$$.

- 因为$$Ax=b$$在原始问题的可行域上, d 总能取到不大于原始问题的极小值.

#### 1.3.2 不等式约束线性规划

#### 1.3.3 一般约束问题

下面我们再说将约束放到目标函数上的另一个好处: **将目标函数转化为原始目标函数的一个下界**:

1. 首先来看等式约束, 因此等式约束中对拉格朗日乘子没有限制.
2. 在不等式约束中, 限制$$\lambda \in R^m_+$$

### 总结

- 使用拉格朗日松弛, 可以把问题进行分解

1. **作拉格朗日松弛并不是要把所有约束都放到目标函数上去的**, 而是有选择性的对约束进行处理.
   - 如何区分难的约束和简单的约束呢?主要依靠个人经验判断.
   - 但是, 一般而言, 选择 linked/coupling constraints, 可以让不可分的问题分解为简单的子问题.
2.

## 参考:

- https://www.zhihu.com/question/58584814
- https://www.cnblogs.com/massquantity/p/10807311.html
- https://zhuanlan.zhihu.com/p/115745075
- https://zhuanlan.zhihu.com/p/103961917
- https://blog.csdn.net/Mr_KkTian/article/details/53750424
- https://www.zybuluo.com/dongxi/note/848084#:~:text=%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E5%AF%B9%E5%81%B6%E6%80%A7%E6%98%AF%E4%B8%80%E7%A7%8D%E5%AF%BB%E6%89%BE%E5%A4%9A%E5%85%83%E5%87%BD%E6%95%B0%E5%9C%A8%E5%85%B6%E8%87%AA%E5%8F%98%E9%87%8F%E5%8F%97%E5%88%B0%E4%B8%80%E4%B8%AA%E6%88%96%E8%80%85%E5%A4%9A%E4%B8%AA%E6%9D%A1%E4%BB%B6%E7%BA%A6%E6%9D%9F%E6%97%B6%E7%9A%84%E6%9E%81%E5%80%BC%E7%9A%84%E6%96%B9%E6%B3%95%E3%80%82%20%E8%BF%99%E7%A7%8D%E6%96%B9%E6%B3%95%E5%8F%AF%E4%BB%A5%E5%B0%86%E4%B8%80%E4%B8%AA%E6%9C%89,%E4%B8%AA%E5%8F%98%E9%87%8F%E5%92%8C%20%E4%B8%AA%E7%BA%A6%E6%9D%9F%E6%9D%A1%E4%BB%B6%E7%9A%84%E6%9C%80%E4%BC%98%E5%8C%96%E9%97%AE%E9%A2%98%E8%BD%AC%E6%8D%A2%E4%B8%BA%E4%B8%80%E4%B8%AA%E8%A7%A3%E6%9C%89%20%E4%B8%AA%E5%8F%98%E9%87%8F%E7%9A%84%E6%96%B9%E7%A8%8B%E7%BB%84%E9%97%AE%E9%A2%98%E3%80%82
- https://blog.csdn.net/CloudInSky1/article/details/122297915
- https://zhuanlan.zhihu.com/p/522590887
- https://zhuanlan.zhihu.com/p/145944142

- **松弛后的问题称为松弛问题(Primal,P)**
