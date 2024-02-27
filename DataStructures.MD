

# Fenwick Tree

    binary indexed tree to get sum on range with update on a single value.
    could be used in sum, oxr, multiplication,...
    you should take care that it works 1 based
    Don't forget `build()`
    complexity: 
    - build : O(n log n)
    - add, prefix sum: O(n log n)

## 1D

``` c++ 

int n;
vector<int> a, bit;
// O(log n)
void add(int index, int toAdd)
{
    if (index == 0)
        return;
    // add the least significant bit to i
    for (int i = index; i <= n; i += (i & -i))
        bit[i] += toAdd;
}
// O(n log n)
void build()
{
    bit = vector<int>(n + 1);
    for (int i = 1; i <= n; i++)
        add(i, a[i]);
}
// O(log n)
ll prefixSum(int index)
{
    ll sum = 0;
    // accumulate the sum and update i by removing its least significant bit
    for (int i = index; i >= 1; i -= (i & -i))
        sum += bit[i];
    return sum;
}
// O(log n)
int rangeSum(int l, int r)
{
    return prefixSum(r) - prefixSum(l - 1);
}
// O(log n)
void update(int index, int value)
{
    add(index, value - a[index]);
}



```

## 2D


``` c++
int n, m;
vector<vector<int>> a, bit;
// O(log n* log m)  
void add(int x, int y, int toAdd)
{
    // handle if x == 0 or y== 0
    // add the least significant bit to i
    for (int i = x; i <= n; i += (i & -i))
        for (int j = y; j <= m; j += (j & -j))
            bit[i][j] += toAdd;
}
// O(n*m log n* log m)  
void build()
{
    bit = vector<vector<int>>(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            add(i, j, a[i][j]);
}
ll prefixSum(int x, int y)
{
    ll sum = 0;
    // accumulate the sum and update i by removing its least significant bit
    for (int i = x; i >= 1; i -= (i & -i))
        for (int j = y; j >= 1; j -= (j & -j))
            sum += bit[i][j];
    return sum;
}
// from point (l1,r1) [at top left corner] to point (l2,r2) [at bottom right corner]
int rangeSum(int l1, int r1, int l2, int r2)
{
    return prefixSum(l2, r2) - prefixSum(l1 - 1, r2) - prefixSum(l2, r1 - 1) + prefixSum(l1 - 1, r1 - 1);
}
void update(int x, int y, int value)
{
    add(x, y, value - a[x][y]);
}

```
