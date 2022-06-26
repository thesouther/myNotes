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

此外, 还可以将困难的约束直接放进目标函数里, 通过引入一个 price, 对违反该约束的函数取值增加一些代价. 这就是"**拉格朗日松弛**"的基本思想.

### 示例 1

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

问题看起来简单了很多.

### 示例 2 选址问题

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
