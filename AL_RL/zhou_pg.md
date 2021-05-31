# 4 周博磊RL-6-策略优化
1. 优势:
   - 收敛性好, 至少是局部最优; 
   - 在高维动作空间更高效, PG输出是向量, value-based只是单值; 
   - 可以学习随机策略, value-based不可以. 比如剪刀石头布游戏, 随机更好.
2. 劣势
   - 经常收敛到局部最优; 
   - 评估策略方差比较大.

## 1. 策略梯度

1. 推导
2. 对于无法求导的性能函数,可以使用Cross Entropy Method (CEM)或者有限微分(Finite Difference)
   
   
   <div style="text-align: center; width: 70%; margin: auto; ">
   <img width=100% src="img/2021_05_31_23_05_35.png">
   <img width=100% src="img/2021_05_31_23_06_06.png">
   </div>
3. score-function, $$\triangledown_\theta \log \pi_\theta(s,a) $$
   score function指的应该是每个样本会对应一个score, 这个score近似衡量了这个样本对于log likelihood的贡献. 最终的log likellihood就是每个样本的score加起来再加上一个常数. 因此, 当score的方差大的时候, 每个样本就提供了关于loglikelihood更多的信息, 所以用MLE估计参数的时候, 得到的参数估计的方差反而越小.
   
   <div style="width: 100%; height:100px; line-height:100px; text-align: center; ">
   <div style="float: right; width:15%; height:100%; ">
   <p>(1)</p>
   </div>
   <div style="float: right; width:80%; height:100%; ">
   <img src="img/2021_05_31_23_11_41.png">
   </div>
   </div>

### 策略举例

<div style="text-align: center; width: 50%; margin: auto; ">
<img width=100% src="img/2021_05_31_23_12_56.png">
<img width=100% src="img/2021_05_31_23_13_13.png">
</div>
