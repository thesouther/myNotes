# ssr服务器

## 1 注册AWS，
- 开启实例，比如Ubuntu18.04，
- 下载`*.pem`，密钥文件。
- 设置安全组，新建规则-> 端口10004-10014->来源选择全部。
- 开启弹性IP，这个如果不用的话会产生费用。如果不经常重启服务器可以先不用。

## 2 登录AWS

1. 生成`.ssh`文件夹,
   
    ```
    ssh-keygen -t rsa -C "你的邮箱"
    ```

2. 把`*.pem`密钥文件放到`C:\Users\你的用户名\.ssh\`文件夹下。
3. 添加`config`文件，不带文件后缀。在文件里加入
    
    ```bash
    Host your_host_name
    HostName 13.250.32.196
    Port 22
    User ubuntu
    IdentityFile C:\Users\你的用户名\.ssh\***.pem
    ```

4. 连接AWS服务器
   
    ```bash
    ssh your_host_name
    ```

## 3 搭建ssr服务

```
wget -N --no-check-certificate https://raw.githubusercontent.com/ToyoDAdoubi/doubi/master/ssr.sh
chmod +x ssr.sh
sudo bash ssr.sh

输入1安装

设置端口号（你的安全组设置的端口号）

设置密码

设置加密方式10

设置协议1

设置混淆1

设置连接设备数限制

设置限速

开始安装，等待安装完成

安装成功！
```

## 4 安装BBR加速

```bash
wget –no-check-certificate https://github.com/teddysun/across/raw/master/bbr.sh
chmod +x bbr.sh
./bbr.sh
 ```

重启服务器，服务器端到此大功告成。