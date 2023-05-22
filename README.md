# Project —— Matrix Multiplication with CUDA

## 1. Tile size 1×1 per thread

在本方法中，每一个thread负责抽取A中的一行和B中的一列进行计算（即一个tile），得到C中的一个元素。注意到矩阵是以row-major的方式储存的，因此代码如下：

```c++
int sum = 0;
for (int i = 0; i < A.width; i++) 
{
    sum += A.elements[idx_y * A.width + i] * B.elements[i * B.width + idx_x];
}
C.elements[idx_y * C.width + idx_x] = sum;
```

运行结果如下：

## 2. Incresing tile size per thread