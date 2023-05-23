# Project —— Matrix Multiplication with CUDA

## 1. Tile size 1×1 per thread

在本方法中，每一个thread负责抽取A中的一行和B中的一列进行计算（即一个tile），得到C中的一个元素，如图所示：

![image.png](https://s2.loli.net/2023/05/22/RIQzjDlUGoudJvC.png)

注意到矩阵是以row-major的方式储存的，代码如下：

```c++
int sum = 0;
for (int i = 0; i < A.width; i++) 
{
    sum += A.elements[idx_y * A.width + i] * B.elements[i * B.width + idx_x];
}
C.elements[idx_y * C.width + idx_x] = sum;
```

运行结果如下：

![image.png](https://s2.loli.net/2023/05/22/OHR1FU9A82mfSMC.png)

该方法的平均运行时间为：3.006406 ms。

## 2. Incresing tile size per thread

本方法中，每一个thread负责的tile数增多，以增强数据的复用性。如图所示：

![image.png](https://s2.loli.net/2023/05/22/4ePwj9tYcQ2EKMA.png)

首先对`Csum`中的元素清零：

```c++
memset(Csum, 0, sizeof(Csum));
```

之后初始化`a_vec`和`b_vec`，并做乘累加运算：

```c++
for (int i = 0; i < TILE_SIZE; i++)
{
    a_vec[i] = A.elements[(tile_row + i) * A.width + k];
    b_vec[i] = B.elements[k * B.width + (tile_col + i)];
}

for (int i = 0; i < TILE_SIZE; i++)
{
    for (int j = 0; j < TILE_SIZE; j++)
    {
        Csum[i][j] += a_vec[i] * b_vec[j];
    }
} 
```

循环结束之后，将`Csum`中的元素进行写回：

```c++
for (int i = 0; i < TILE_SIZE; i++)
{
    for (int j = 0; j < TILE_SIZE; j++)
    {
        C.elements[(tile_row + i) * C.width + (tile_col + j)] = Csum[i][j];
    }
}
```

运行结果如下(`TILE_SIZE=2`)：

![image.png](https://s2.loli.net/2023/05/23/JwB6jAzhTfyREo9.png)

改变`TILE_SIZE`的大小，并对运行时间进行统计：

| tile size | time(ms) |
| --------- | -------- |
|     1     | 3.282611 |
|     2     | 2.271091 |
|     4     | 2.107053 |
|     8     | 2.077536 |
|    16     | 7.368474 |
|    32     | 25.815271|
|    64     | 83.601498|

随`TILE_SIZE`的增加，运行时间先减小后增大（在8左右达到最小值，随后急剧增大）。

1. 注意到此处A和B均存储在global memory中，访存开销较大。当tile size从1增加到8时，对A中特定行/B中特定列的访存数减小，计算密度（MACS操作的数目与访存操作的比例）增大；
2. 考虑到每一个warp中含有32个线程，最多含有256个向量寄存器，其中每个寄存器可以存放32个32位元素，均摊在每一个线程上，最多能有256个元素能被存放于寄存器中。而当tile size大于16时，光`C_sum`一项中就含有256个32位int元素，多余的元素只能存放在L1 cache中，造成较大的开销。此外寄存器使用率接近100%时，可能会造成active warp数减小、bank conflict等。

## 3. Optimization using shared memory

注意到在上例中，一段重复的代码可能会被多个thread所用到，故考虑将其载入shared memory中。如图所示：

![image.png](https://s2.loli.net/2023/05/22/tnwG183rLXzkTAb.png)

首先在计算之前，要确保shared memory正确初始化：

```c++
As[row][col] = GetElement(Asub, row, col);
Bs[row][col] = GetElement(Bsub, row, col);
__syncthreads();
```

之后从shared memory中读取数据进行乘累加运算：

```c++
for (int e = 0; e < BLOCK_SIZE; ++e)
{
    Cvalue += As[row][e] * Bs[e][col];
}
__syncthreads();
```

最后写回结果：

```c++
SetElement(Csub, row, col, Cvalue);
```

运行结果如下(`TILE_SIZE=16`)：

![image.png](https://s2.loli.net/2023/05/23/dUXTZPh62fRVj1p.png)

改变`TILE_SIZE`的大小，并对运行时间进行统计：

| tile size | time(ms) |
| --------- | -------- |
|     1     | 62.395489|
|     2     | 10.850624|
|     4     | 2.765997 |
|     8     | 2.107232 |
|    16     | 1.849939 |
|    32     | 1.928793 |

首先，引入shared memory之后，每一个thread无需像上一种方法那样，维护自己的`C_sum`数组，这样寄存器的使用状况得到了缓解；当tile size增大时，数据的复用性增强，只要不超过线程块所支持的最大线程数，执行时间就会一直减小。

## Reference

[^1]: [Achieved Occupancy](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm)

[^2]: [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)