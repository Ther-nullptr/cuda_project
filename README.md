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

运行结果如下：



改变`TILE_SIZE`的大小，并对运行时间进行统计：

| tile size | time(ms) |
| --------- | -------- |
|           |          |


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

运行结果如下：



改变`TILE_SIZE`的大小，并对运行时间进行统计：

| tile size | time(ms) |
| --------- | -------- |
|           |          |