# ! [rejected]master -> master (non-fast-forward)

## 问题

```
! [rejected]master -> master (non-fast-forward)
```

# 解决

```
git pull --rebase origin master
git push -u origin master
```