# 7-凸优化

<div style="text-align: center; width: 80%; margin: auto; ">
<img width=100% src="img/2022_06_19_21_39_50.png">
</div>

$$
$$

## 基础概念

- 优化: 从⼀个可⾏解的集合中, 寻找最优的元素
  - 线性规划/非线性规划
  - 凸优化/非凸优化
  - 光滑/非光滑
  - 单目标优化/多目标优化, (多目标优化往往无法同时使得多个目标函数最小，需要进行折中选择， 或者将多个目标进行加权求和)
- 凸优化: 如果一个问题是凸优化问题, 那么其目标函数是凸函数, 约束为凸集(约束由若干个凸函数组成)
  - 线性规划一定是一个凸规划

|       名称        |        英文        |                                                                                                定义                                                                                                |                                                                                      解释                                                                                      |
| :---------------: | :----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 过任意两点的直线  |                    |                                                                          $$ f({x}) = θx_1 + (1 − θ) x_2 ,\theta \in R $$                                                                           |                                                                                                                                                                                |
| 过任意两点的线段  |                    |                                                                        $$ f({x}) = θx_1 + (1 − θ) x_2 ,\theta \in [0,1] $$                                                                         |                                                                                                                                                                                |
| n 维空间的子空间  |                    |                                                 $$\alpha, \beta\in V, k\in R\Rightarrow \alpha+\beta\in V, k\alpha \in V$$, 则 V 为$$R^n$$的子空间                                                 | $$子空间: R^n的非空子集V, 且对加法和数乘运算封闭$$. 对减法肯定也封闭. <br>n 维空间的子空间一定包含零向量. <br> 过原点的平面或直线, 以原点为起点的所有向量为$$R^3$$空间的子空间 |
|     线性函数      |                    |                                                                          $$f(\alpha x+\beta y) = \alpha f(x)+\beta f(y)$$                                                                          |                                                                                                                                                                                |
|      凸函数       |                    |                                                                        $$f(\alpha x+\beta y) \leq \alpha f(x)+\beta f(y)$$                                                                         |                                                                                                                                                                                |
|      仿射集       |    Affine Sets     |                                                                  一个集合 C 中，连接任意两点的直线也在该集合中，则该集合为仿射集                                                                   |                                                              直线/二维空间都是仿射集; 线段、闭合图形不是仿射集。                                                               |
|     仿射组合      | affine combination | 设集合 C 中的 k 个点, $$x_{1}, \cdots, x_{k} \in C; \theta_{1}, \cdots, \theta_{k} \in \mathbb{R}, \theta+\cdots+\theta_{k}=1$$, 则$$\theta_{1} x_{1}+\cdots+\theta_{k} x_{k}$$为 k 个点的仿射组合 |                                                            **若集合 C 中 k 个点的仿射组合也在 C 中, 则 C 为仿射集**                                                            |
|      仿射包       |    affine hull     |                            $$\text { aff } C=\left\{\theta_{1} x_{1}+\cdots+\theta_{k} x_{k} \mid x_{1}, \ldots, x_{k} \in C, \theta_{1}+\cdots+\theta_{k}=1\right\}$$                             |                                                     集合 C 中元素的所有仿射组合, (包含集合 C 的最小仿射集)称为 C 的仿射包                                                      |
| 与 C 相关的子空间 |                    |                                                                C 是仿射集,$$ V = C-x_0 = \{x-x_0 \| x \in C\}, \forall x_0 \in C $$                                                                |    该式表示对仿射集 C 的平移, 过原点, 具有更好的性质. <br> $$\forall V_{1}, V_{2} \in V ,   \forall \alpha, \beta \in \mathbb{R} , 使得  \alpha V_{1}+\beta V_{2} \in V $$     |
|       凸集        |     Convex Set     |                        $$C \text { is a Convex Set } \Leftrightarrow \forall \theta x_{1}+(1-\theta) x_{2} \in C, x_{1}, x_{2} \in C \forall \theta \quad \theta \in[0,1]$$                        |                                             一个集合是凸集，当属于该集合的任意两点之间的线段仍然在该集合内。 <br>仿射集一定是凸集                                              |
|      凸组合       | convex combination |                                             $$\theta_{1} x_{1}+\cdots+\theta_{k} x_{k},  \theta_{1}+\cdots+\theta_{k}=1, \theta_{i} \ge 0, i=1,...k $$                                             |                                                               $$C 为凸集 \Leftrightarrow 任意元素凸组合 \in C $$                                                               |
|       凸包        |    convex hull     |                           $$Conv  C=\left\{\theta_{1} x_{1}+\cdots+\theta_{k} x_{k} \mid x_i \in C, \theta_i \ge 0, i=1,...k,  \theta_{1}+\cdots+\theta_{k}=1\right\}$$                            |                                                 集合 C (不一定是凸集) 中元素的所有凸组合, (包含集合 C 的最小凸集)称为 C 的凸包                                                 |
|        锥         |                    |                                                      $$C  是锥  \Leftrightarrow \quad \forall x \in C, \theta \geq 0 , 有  \theta x \in C $$                                                       |                                                                              锥一定是过原点的集合                                                                              |
|       凸锥        |    Convex Cone     |                            $$C  是凸锥  \Leftrightarrow \quad \forall x_{1}, x_{2} \in C, \theta_{1}, \theta_{2} \geq 0 , 有  \theta_{1} x_{1}+\theta_{2} x_{2} \in C$$                            |                                                                                                                                                                                |
|      锥组合       | conic combination  |                                                     $$\theta_{1} x_{1}+\cdots+\theta_{k} x_{k} \quad \theta_{1}, \cdots, \theta_{k} \geq 0 $$                                                      |                                                                                                                                                                                |
|      凸锥包       |     conic hull     |                                      $$\{\theta_{1} x_{1}+\cdots+\theta_{k} x_{k} \mid x_{1}, \cdots, x_{k} \in C, \theta_{1}, \cdots, \theta_{k} \geq 0\}$$                                       |                                                                                                                                                                                |
|                   |                    |                                                                                                                                                                                                    |                                                                                                                                                                                |

比较

|   名称   |                                                       比较                                                        |
| :------: | :---------------------------------------------------------------------------------------------------------------: |
| 仿射组合 |                 $$\forall \theta_{1}, \cdots, \theta_{k}, \quad \theta_{1}+\cdots+\theta_{k}=1$$                  |
|  凸组合  | $$\forall \theta_{1}, \cdots, \theta_{k} \quad \theta_{1}+\cdots+\theta_{k}=1, \theta_{1}, \cdots, \theta_{k}>0$$ |
| 凸锥组合 |               $$\forall \theta_{1}, \cdots, \theta_{k} \quad \theta_{1}+\cdots+\theta_{k} \geq 0$$                |

### 关键定理

- 仿射集
  - 任意线性方程组的解集都是仿射集
  - 任意一个仿射集, 都可以写成一个线性方程组的解集
  - 与 C 相关的子空间, 性质更好的子空间, 一定是过原点的
- 仿射集是凸集的一个特例
- 单个点, 一定是仿射集, 凸集, 不是凸锥集
- 空集, 是仿射集, 凸集, 凸锥集
- 直线
  - 过原点: 仿射集, 凸集, 凸锥集
  - 不过原点: 仿射集, 凸集, 不是凸锥集
