# LCA
- build in $O(nlogn)$.
- Query in $O(logn)$.
```cpp
const int L = 25;
ll up[N][L], tin[N], tout[N], timer;

void dfs(int v, int p)
{
    tin[v] = ++timer;
    up[v][0] = p;
    for (int i = 1; i < L; ++i)
        up[v][i] = up[up[v][i-1]][i-1];

    for (int u : adj[v]) {
        if (u != p)
            dfs(u, v);
    }

    tout[v] = ++timer;
}

bool is_ancestor(int u, int v)
{
    return tin[u] <= tin[v] && tout[u] >= tout[v];
}

int lca(int u, int v)
{
    if (is_ancestor(u, v))
        return u;
    if (is_ancestor(v, u))
        return v;
    for (int i = L - 1; i >= 0; --i) {
        if (!is_ancestor(up[u][i], v))
            u = up[u][i];
    }
    return up[u][0];
}

int main()
{

	dfs(root, root);
}
```
# Centroid Decomposition

### 1st method
- build in $O(nlogn^2)$.
- update and query in $O(logn)$.
```cpp
set<ll>g[N]; // it's not vector<vector<ll>>!
ll dad[N], sub[N];

ll getSize(ll u, ll p) {	// O(n)
    sub[u] = 1;

    for (auto v : g[u])
        if (v != p) sub[u] += getSize(v, u);

    return sub[u];
}

ll getCentroid(ll u, ll p, ll n) {	// O(n)
    for (auto v : g[u])
        if (v != p and sub[v] > n/2) return getCentroid(v, u, n);

    return u;
}

void build(ll u, ll p) {	// O(nlogn)
    ll sz = getSize(u, p); // find the size of each subtree
    ll centroid = getCentroid(u, p, sz); // find the centroid
    if (p == -1) p = centroid; // dad of root is the root itself
    dad[centroid] = p;

    // for each tree resulting from the removal of the centroid
    for (auto v : g[centroid])
        g[centroid].erase(v), // remove the edge to disconnect
        g[v].erase(centroid), // the component from the tree
        build(v, centroid);
}

int main()
{

	build(1, -1);
}
```

### 2nd method (more optimized)
- worked using `vis` array but not erase edges.
- build in $O(nlogn)$.
- update and query in $O(logn)$.

```cpp
int getSize(int u, int p) {
  if (vis[u]) return 0;
    sub[u] = 1;
 
    for (auto v : g[u])
        if (v != p && vis[v] == 0) sub[u] += getSize(v, u);
 
    return sub[u];
}
 
int getCentroid(int u, int p, int sz) {
    for (auto v : g[u])
        if (v != p and sub[v] > n/2 and vis[v] == 0) return getCentroid(v, u, sz);
 
    return u;
}
 
void build(int u, int p) {
    int sz = getSize(u, 0); // find the size of each subtree
    int centroid = getCentroid(u, -1, sz); // find the centroid
    dad[centroid] = p;
    vis[centroid] = 1;
    // for each tree resulting from the removal of the centroid
    for (auto v : g[centroid])
    {
        // g[v].erase(centroid); // the component from the tree
        if (vis[v] == 0)build(v, centroid);
    }
    // g[centroid].clear(); // remove the edge to disconnect
}
 
void update(int v)
{
    // ans[v] = 0;
    int u = v;
    while (u != -1)
    {
        ans[u] = min(ans[u], dis(u, v));
        u = dad[u];
    }
}
 
int getAns(int v)
{
    int ret = 1e9;
    int u = v;
    while (u != -1)
    {
        ret = min(ret, ans[u] + dis(u, v));
        u = dad[u];
    }
    return ret;
}
```
