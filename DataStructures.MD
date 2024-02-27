

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
## farest postion bigger that or equal to k from l to r-1
```cpp
int calc(int l, int r, int x, int lx, int rx, int k)
  {
    if (rx <= l || lx >= r || arr[x].first < k)
      return -1;
    if (rx - lx == 1)
      return arr[x].second;
    int m = (lx + rx) / 2;
    int res = -1;
    res = calc(l, r, 2 * x + 2, m, rx, k);
    if (res == -1)
      res = calc(l, r, 2 * x + 1, lx, m, k);
    return res;
  }
  int calc(int l, int r, int k)
  {
    return calc(l, r, 0, 0, size, k);
  }
```
## 2D
``` cpp
ll a[1001][1001];    
ll arr[4000][4000];
        
struct segtree{
    int size;
    void init(int n){
        size=1;
        while(size<n)size*=2;
        // arr.resize(size,vector<ll>(size,0));
        // for(int i=0;i<size*2;i++)arr.push_back(NEUel);
    }
    void build_y(int x,int lx,int rx,int y,int ly,int ry)
    {
        if (ry-ly == 1) {
        if (lx == rx-1)
            arr[x][y] = a[lx][ly];
        else
            arr[x][y] = arr[x*2+1][y] + arr[x*2+2][y];
    } else {
        int my = (ly + ry) / 2;
        build_y(x, lx, rx, y*2+1, ly, my);
        build_y(x, lx, rx, y*2+2, my, ry);
        arr[x][y] = arr[x][y*2+1] + arr[x][y*2+2];
    }
    }
    void build_x(int x,int lx,int rx)
    {
        if(rx-lx!=1){
            int m=(lx+rx)/2;
            build_x(2*x+1,lx,m);
            build_x(2*x+2,m,rx);
        }
        build_y(x,lx,rx,0,0,size);
    //arr[x]=merge(arr[2*x+1],arr[2*x+2]);
    }
    void build()
    {
        build_x(0,0,size);
    }
    void update_y(int x, int lx, int rx, int y, int ly, int ry, int a, int b, int new_val) {
        if (ly == ry-1) {
            if (lx == rx-1)
            {
                arr[x][y] = (arr[x][y]+1)%2;
                // cout<<arr[x][y]<<'\n';
            }
            else
                arr[x][y] = arr[x*2+1][y] + arr[x*2+2][y];
        } else {
            int my = (ly + ry) / 2;
            if (b < my)
                update_y(x, lx, rx, y*2+1, ly, my, a,b, new_val);
            else
                update_y(x, lx, rx, y*2+2, my, ry, a, b, new_val);
            arr[x][y] = arr[x][y*2+1] + arr[x][y*2+2];
        }
    }




    void update_x(int x, int lx, int rx, int a, int b, int new_val) {
    if (lx != rx-1) {
        int mx = (lx + rx) / 2;
        if (a < mx)
            update_x(x*2+1, lx, mx, a, b, new_val);
        else
            update_x(x*2+2, mx, rx, a, b, new_val);
    }
    update_y(x, lx, rx, 0, 0, size, a, b, new_val);
    }
    void update_x(int a,int b,int val=0){
        update_x(0,0,size,a,b,val);
    }

    int sum_y(int x, int y, int tly, int try_, int ly, int ry) {
        if (tly>=ry||try_<=ly) 
            return 0;
        if (ly <= tly && try_ <= ry)
            return arr[x][y];
        int tmy = (tly + try_) / 2;
        return sum_y(x, y*2+1, tly, tmy, ly, ry)
            + sum_y(x, y*2+2, tmy, try_, ly, ry);
    }
    int sum_x(int x, int tlx, int trx, int lx, int rx, int ly, int ry) {
        if (tlx>=rx||trx<=lx)
            return 0;
        if (lx <= tlx && trx <= rx)
            return sum_y(x, 0, 0,size, ly, ry);
        int tmx = (tlx + trx) / 2;
        return sum_x(x*2+1, tlx, tmx, lx, rx, ly, ry)
            + sum_x(x*2+2, tmx, trx, lx, rx, ly, ry);
    }
    int sum(int a,int b,int c,int d)
    {
        return sum_x(0,0,size,a,b,c,d);
    }
};    

void fn()
{
    cin>>n>>m;
    segtree st;
    st.init(n);
    string s;
    for(int i=0;i<n;i++)
    {
        cin>>s;
        for(int j=0;j<n;j++)
        {
            a[i][j]=(int)(s[j]=='*');
            //cout<<a[i][j]<<" ";
        }
        cout<<'\n';
    }
    // cout<<1;
    st.build();
    // return;
    while(m--)
    {
        int c;
        ll a,b;
        cin>>c>>a>>b;
        if(c==2)
        {
            ll d,f;
            cin>>d>>f;
            cout<<st.sum(a-1,d,b-1,f)<<'\n';
        }
        else
        {
            st.update_x(a-1,b-1);
        }
    }
}
```
### Segment with Pashka
#### inversions 2

```cpp
#include <bits/stdc++.h>

using namespace std;

#define IO ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
#define ll long long

const int N = 2e5 + 5;
const ll MOD = 1e9 + 7;

// This problem is the reversed version of the previous one. 
// There was a permutation pi of n elements, 
// for each i we wrote down the number ai the number of j such that j<i and pj>pi 
// Restore the original permutation for the given ai.

ll n, a[N], seg[4 * N], pre, ans[N];
 
void build(ll x, ll lx, ll rx)
{
    if (lx == rx)return void(seg[x] = 1);
    ll mx = (lx + rx) >> 1;
    build(2 * x, lx, mx);
    build(2 * x + 1, mx + 1, rx);
    seg[x] = seg[2 * x] + seg[2 * x + 1];
}
 
void update(ll x, ll lx, ll rx, ll pos)
{
    if (pos < lx || pos > rx)return;
    if (lx == rx)return void(seg[x] = 0);
    ll mx = (lx + rx) >> 1;
    update(2 * x, lx, mx, pos);
    update(2 * x + 1, mx + 1, rx, pos);
    seg[x] = seg[2 * x] + seg[2 * x + 1];
}
 
ll query(ll x, ll lx, ll rx, ll k)
{
    if (lx == rx)return lx;
    ll mx = (lx + rx) >> 1;
    if (seg[2 * x + 1] >= k)return query(2 * x + 1, mx + 1, rx, k);
    return query(2 * x, lx, mx, k - seg[2 * x + 1]);
}
 
int main()
{
    IO
    cin >> n;
    build(1, 1, n);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
    }
    for (int i = n; i >= 1; i--)
    {
       ans[i] = query(1, 1, n, a[i] + 1);
       update(1, 1, n, ans[i]);
    }
    for (int i = 1; i <= n; i++)
    {
        cout << ans[i] << ' ';
    }
    return 0;
}
```

#### Nested Segments

```cpp
#include <bits/stdc++.h>

using namespace std;

#define IO ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
#define ll long long

const int N = 2e5 + 5;
const ll MOD = 1e9 + 7;

// Given an array of 2n numbers, 
// each number from 1 to n in it occurs exactly twice. 
// We say that the segment y is nested inside the segment x if both occurrences 
// of the number y are between the occurrences of the number x Find for each segment i 
// how many segments that are nested inside it.

ll n, R[N], seg[4 * N], x, ans[N];
vector<ll>L;
 
void update(ll x, ll lx, ll rx, ll pos, ll value)
{
    if (pos < lx || pos > rx)return;
    if (lx == rx)return void(seg[x] = value);
    ll mx = (lx + rx) >> 1;
    update(2 * x, lx, mx, pos, value);
    update(2 * x + 1, mx + 1, rx, pos, value);
    seg[x] = seg[2 * x] + seg[2 * x + 1];
}
 
ll query(ll x, ll lx, ll rx, ll r)
{
    if (r < lx)return 0;
    if (rx <= r)return seg[x];
    ll mx = (lx + rx) >> 1;
    return query(2 * x, lx, mx, r) + query(2 * x + 1, mx + 1, rx, r);
}
 
int main()
{
    scanf("%lld", &n);
    for (int i = 1; i <= 2 * n; i++)
    {
        scanf("%lld", &x);
        if (R[x] == -1)R[x] = i, update(1, 1, 2 * n, i, 1);
        else L.push_back(x), R[x] = -1;
    }
    for (auto l : L)
    {
        ans[l] =  query(1, 1, 2 * n, R[l] - 1);
        update(1, 1, 2 * n, R[l], 0);
    }
    for (int i = 1; i <= n; i++)
    {
        printf("%lld ", ans[i]);
    }
    return 0;
}
```
#sparse table
```
complexity: O(nlogn) build and O(1) to answer queries for max ,min queries O(logn).
```
```cpp
long long st[K + 1][MAXN];
//init
std::copy(array.begin(), array.end(), st[0]);

for (int i = 1; i <= K; i++)
    for (int j = 0; j + (1 << i) <= N; j++)
        st[i][j] = st[i - 1][j] + st[i - 1][j + (1 << (i - 1))];

//compute sum from l to r
long long sum = 0;
for (int i = K; i >= 0; i--) {
    if ((1 << i) <= R - L + 1) {
        sum += st[i][L];
        L += 1 << i;
    }
}

// log precalculation
int lg[MAXN+1];
lg[1] = 0;
for (int i = 2; i <= MAXN; i++)
    lg[i] = lg[i/2] + 1;

//Alternatively, log can be computed on the fly in constant space and time:
// C++20
#include <bit>
int log2_floor(unsigned long i) {
    return std::bit_width(i) - 1;
}

// pre C++20
int log2_floor(unsigned long long i) {
    return i ? __builtin_clzll(1) - __builtin_clzll(i) : -1;
}


//Afterwards we need to precompute the Sparse Table structure. This time we define $f$ with $f(x, y) = \min(x, y)$.

int st[K + 1][MAXN];

std::copy(array.begin(), array.end(), st[0]);

for (int i = 1; i <= K; i++)
    for (int j = 0; j + (1 << i) <= N; j++)
        st[i][j] = min(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);

//And the minimum of a range $[L, R]$ can be computed with:

int i = lg[R - L + 1];
int minimum = min(st[i][L], st[i][R - (1 << i) + 1]);


int lg[N+1];
int table[6][17][N];
void build()
{
    lg[1] = 0;
    for (int i = 2; i <= n; i++)
    lg[i] = lg[i/2] + 1;
    
    for(int i = 0; i < n; i++)
      for(int j = 0; j < m; j++)
        table[j][0][i] = arr[i][j];

    for(int k=0;k<m;k++)
      for (int i = 1; i <= 16; i++)
          for (int j = 0; j + (1 << i) <= n; j++)
              table[k][i][j] = max(table[k][i - 1][j], table[k][i - 1][j + (1 << (i - 1))]);
}
int get(int l, int r,int indx)
{
  if(l>r)return 0;
  int i = lg[r - l + 1];
  int ret = max(table[indx][i][l], table[indx][i][r - (1 << i) + 1]);
  return ret;
}

```
# Mo algorithm

```
   0 based
   Don't forget to sort 
   check for overflow
  complexity: O((q+n) * sqrt(n))
```

``` c++
int n, m, len, a[N], fr1[N], fr2[N], ans[N], res;
struct query
{
    int l, r, idx;
} q[N];
bool cmp(query one, query two)
{
    if (one.l / len != two.l / len)
        return one.l < two.l;
    if ((one.l / len) & 1)
        return two.r < one.r;
    else
        return one.r < two.r;
}
void add(int i)
{
    int cur = ++fr1[a[i]];
    fr2[cur - 1]--;
    fr2[cur]++;
    res = max(res, cur);
}
void remove(int i)
{
    int cur = --fr1[a[i]];
    fr2[cur + 1]--;
    fr2[cur]++;
    if (fr2[res] == 0)
        res--;
}
void solve()
{
    scanf("%d %d", &n, &m);
    len = (sqrt(n)) + 1;
    for (int i = 0; i < n; i++)
        cin >> a[i];
    for (int i = 0; i < m; i++)
    {
        int l, r;
        scanf("%d %d", &q[i].l, &q[i].r);
        q[i].idx = i;
    }
    sort(q, q + m, cmp);
    int curl = 1, curr = 0;
    for (int i = 0; i < m; ++i)
    {
        while (curl < q[i].l)
            remove(curl++);
        while (curl > q[i].l)
            add(--curl);
        while (curr < q[i].r)
            add(++curr);
        while (curr > q[i].r)
            remove(curr--);
        ans[q[i].idx] = res;
    }
    for (int i = 0; i < m; i++)
        printf("%d\n", ans[i]);
}

```

# square root decomposition

```
  division operation has high cost. save it in a variable.
  complexity : O(sqrt(n)*O(operation time)
```

``` c++
ll n, sqr, q;
void solve() {
    cin >> n;       sqr = (int)ceil(sqrt(n));
    vector<ll>a(n), block(sqr);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        block[i / sqr] += a[i];
    }
    cin >> q;
    while (q--)
    {
        ll t, l, r, sum = 0, idx, val;
        cin >> t;
        if (t == 1) {//answer query
            cin >> l >> r;
            --l, --r;
            for (int i = l; i <= r;) {
                if (i % sqr == 0 and i + sqr - 1 <= r) {// block inside the range 
                    sum += block[i];
                    i += sqr;
                }
                else//brute force movement
                    sum += a[i++];
            }
        }
        else {//update
            cin >> idx >> val;
            idx--;
            block[idx / sqr] -= a[idx];//undo
            a[idx] = val;//update array
            block[idx / sqr] += a[idx];//do
        }
    }
}

```
