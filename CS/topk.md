# 0 CS-TOP-K 问题

## 1 基本解法

全局排序, 时间复杂度$$O(n log n)$$
局部排序, 冒泡排序, $$n*k$$

## 2 快排-Partition 算法

注意是从小到大排序的.

```python
def partition(arr, l, r):
    i = l - 1
    pivot = arr[r]
    for j in range(l, r):
        if arr[j] >= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[r] = arr[r], arr[i + 1]
    return i + 1


def topk(arr, l, r, k):
    pi = partition(arr, l, r)
    if pi + 1 == k:
        return arr[pi]
    elif pi + 1 > k:
        return topk(arr, l, pi - 1, k)
    else:
        return topk(arr, pi + 1, r, k - pi + 1)


arr = [0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8, 9]
n = len(arr)
k = 4
ans = topk(arr, 0, n - 1, k)
print(ans)

```

## 堆排序

**要用小顶堆, O(nlogK)**.

### 1.使用内置 heapq 库, 是最小堆

```python
import heapq

def topk(arr, n, k):
    if k > n:
        return
    hq = []
    for e in arr:
        heapq.heappush(hq, e)
        if len(hq) > k:
            heapq.heappop(hq)
    return hq[0]

arr = [0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8, 9]
n = len(arr)
k = 4
ans = topk(arr, n, k)
print(ans)

```

### 2.自建堆

堆操作: `初始化堆, 向上调整(插入),向下调整(删除堆顶)`

```python
parent = (i-1) // 2    # 取整
left = 2 * i + 1
right = 2 * i + 2
```

实现如下

```python
class MinHeap:
    def __init__(self, arr=[]):
        self.arr = arr
        self._count = 0

    def get_top(self):
        if self._count <= 0:
            raise Exception('The heap is empty!')
        return self.arr[0]

    def __len__(self):
        return self._count

    def add(self, value):
        self.arr.append(value)
        self._count += 1
        self._siftup(self._count - 1)

    def _siftup(self, i):
        if i > 0:
            parent = (i - 1) // 2
            if self.arr[parent] > self.arr[i]:
                self.arr[parent], self.arr[i] = self.arr[i], self.arr[parent]
                self._siftup(parent)

    def pop(self):
        if self._count <= 0:
            raise Exception('The heap is empty!')
        value = self.arr[0]
        self._count -= 1
        self.arr[0] = self.arr[self._count]
        self.arr.pop(-1)
        self._siftdown(0)
        return value

    def _siftdown(self, i):
        count = self._count
        if i < count:
            lc = 2 * i + 1
            rc = 2 * i + 2
            if lc < count and rc < count:
                if self.arr[lc] <= self.arr[rc] and self.arr[lc] <= self.arr[i]:
                    self.arr[lc], self.arr[i] = self.arr[i], self.arr[lc]
                    self._siftdown(lc)
                elif self.arr[lc] >= self.arr[rc] and self.arr[rc] <= self.arr[i]:
                    self.arr[rc], self.arr[i] = self.arr[i], self.arr[rc]
                    self._siftdown(rc)

            elif lc < count and rc >= count:
                if self.arr[lc] <= self.arr[i]:
                    self.arr[lc], self.arr[i] = self.arr[i], self.arr[lc]
                    self._siftdown(lc)


def topk(arr, n, k):
    if k > n:
        return
    hq = MinHeap([])
    for e in arr:
        hq.add(e)
        if len(hq) > k:
            hq.pop()
        print(hq.arr)
    return hq.get_top()


arr = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8, 9]
n = len(arr)
k = 4
ans = topk(arr, n, k)
print(ans)
```
