# 上传大文件出错

## git上传大文件后出错

```bash
remote: error: GH001: Large files detected. You may want to try Git La
```

## 解决

1. 首先根据错误信息找到大文件路径，比如` models/BasicCNN-10-epochs-0.0001-LR-STAGE1`

2. 然后修改commit记录

    ```
    git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch models/BasicCNN-10-epochs-0.0001-LR-STAGE1' --prune-empty --tag-name-filter cat -- --all
    ```

    如果出错

    ```
    Cannot rewrite branches: Your index contains uncommitted changes.
    ```

    解决：

    ```
    git stash
    ```
3. 重新push
   ```
   git push origin master
   ```

4. 清理空间
   ```
    rm -rf .git/refs/original/
    git reflog expire --expire=now --all
    git gc --prune=now
   ```