# Windows安装gym box2d不能用

```
import gym
env = gym.make(id='CarRacing-v0')
```

报错, 没有安装box2d模块

```
AttributeError: module 'gym.envs.box2d' has no attribute 'CarRacing'
```


## 解决

1. 安装siwg. 不能使用pip安装，用anaconda3

```
conda install swig
```

2. 再执行命令安装box2d

```
pip install box2d-py
```