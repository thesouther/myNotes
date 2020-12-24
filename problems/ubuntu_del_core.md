# ubuntu删除与安装系统内核

## 1. 内核安装

1. 搜索可安装的内核版本，使用命令：
   
    ```
    apt-cache  search linux|grep linux-image
    ```

2. 选择所需要的内核版本进行安装，安装内核需要安装image和header，例如：
   
    ```
    apt-get install linux-image-4.4.0-58-generic linux-headers-4.4.0-58-generic
    ```

3. 重启，按ESC进入选择菜单，选择高级选项，选择所需要的内核版本启动系统

## 2. 内核卸载

当我们安装软件时，如果boot空间已满系统会报：

```
gzip: stdout: No space left on device
E: mkinitramfs failure cpio 141 gzip 1
```

如使用如下命令查看boot空间，可发现boot空间已满，这时就需要卸载多余的内核。

```
df -h
```

卸载方法：

1. 首先查看当前使用的内核版本：

```
uname -a
```

2. 正在使用的内核是无法删除的，我们可以删除其他内核.
   查询目前系统中存在的内核版本：

    ```
    dpkg --get-selections | grep linux-image
    ```

3. 使用以下命令卸载当前未使用的多余内核：

    ```
    sudo apt-get remove linux-image-unsigned-4.15.0-107-generic linux-hearders-4.15.0-107-generic
    ```

4. 这时再查看目前系统中存在的内核：

    ```
    dpkg --get-selections | grep linux-image
    ```

    会发现卸载的内核变成啦deinstall

5. 但这个时候还没有完全删除干净，使用命令：

    ```
    sudo dpkg -P linux-image-unsigned-4.15.0-107-generic linux-hearders-4.15.0-107-generic
    ```

    再使用查询命令，会发现其余内核被彻底删除了，boot空间也被释放出来啦