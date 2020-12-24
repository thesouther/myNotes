# ubuntu自动更新导致nvidia驱动不能用

ubuntu自动更新经常导致nvidia驱动不能用， 因为内核更新了。

1. 先查看一下显卡驱动还在不在
    ```
    nvidia-smi
 
    此时显示

    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver  
    ```
2. 查看内核
    ```
    uname -r
    ```
3. 重启系统，进入ubuntu advance option, 选择之前的内核版本
4. **在软件更新里关闭自动更新**
5. [删除不需要的内核.](./ubuntu_del_core.md)