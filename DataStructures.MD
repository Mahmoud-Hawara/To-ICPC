

# Fenwick Tree
```
 binary indexed tree to get sum on range with update on a single value.
 could be used in sum, oxr, multiplication,...
 you should take care that it works 1 based
 Don't forget `build()`

complexity: 
- build : O(n log n)
- add, prefix sum: O(n log n)
```
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

# segtree 
```
-the segmant operation is from l to r-1 `[l,r[` 
-changing the merge for different operations  
complexity: 
- build : O(n)
- add: O(log n)
```
```cpp
struct segtree
{
  int size;
  ll mx = 2e16;
  vector<pair<ll, ll>> minn;
  vector<int> op;
  pair<ll, ll> merge(pair<ll, ll> a, pair<ll, ll> b)
  {
    if (a.first <= b.first)
      return a;
    else
      return b;
  }
  void lazy(int x, int lx, int rx)
  {
    if (op[x] && lx != rx - 1)
    {
      op[2 * x + 1] += op[x];
      op[2 * x + 2] += op[x];
      int z = min(40, op[x]);
      if (minn[2 * x + 1].second)
      {
        minn[2 * x + 1].first /= (1ll << z);
      }
      if (minn[2 * x + 2].second)
      {
        minn[2 * x + 2].first /= (1ll << z);
      }
      op[x] = 0;
    }
  }
  void init(int n)
  {
    size = 1;
    while (size < n)
      size *= 2;
    op.assign(size * 2, 0);
    minn.assign(size * 2, {mx, 0ll});
  }
  void set(int i, ll v, int x, int lx, int rx, ll z)
  {
    if (rx - lx == 1)
    {
      if (minn[x].first > v)
      {
        minn[x].first = v;
        minn[x].second = z;
      }
      return;
    }
    lazy(x, lx, rx);
    int m = (lx + rx) / 2;
    if (i < m)
    {
      set(i, v, 2 * x + 1, lx, m, z);
    }
    else
      set(i, v, 2 * x + 2, m, rx, z);
    minn[x] = merge(minn[2 * x + 1], minn[2 * x + 2]);
  }
  void set(int i, pair<ll, ll> x)
  {
    set(i, x.first, 0, 0, size, x.second);
  }
  pair<ll, ll> fmin(int l, int r, int x, int lx, int rx)
  {
    if (lx >= r || l >= rx)
    {
      return {mx, 0ll};
    }
    if (l <= lx && rx <= r)
    {
      return minn[x];
    }
    lazy(x, lx, rx);
    int m = (lx + rx) / 2;
    return merge(fmin(l, r, 2 * x + 1, lx, m), fmin(l, r, 2 * x + 2, m, rx));
  }
  pair<ll, ll> fmin(int l, int r)
  {
    return fmin(l, r, 0, 0, size);
  }
  void daypassed()
  {
    op[0] += 1;
  }
};
```
