# 0 面试-RL相关

## AlphaStar

1. 计算优势函数, GAE与UPGO

   - GAE和UPGO都关⼼心如何将多步以后未来的信息纳⼊入现在的 Advantage估计中. GAE使⽤用Soft的形式通过λ项控制未来信息和现 在信息的平衡(即偏差和⽅方差的平衡). UPGO则使⽤用Hard的形式 直接将未来乐观的step的信息纳⼊入Advantage.

2. 计算状态值V

   - TD($$\lambda$$)
