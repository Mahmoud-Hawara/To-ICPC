# LCA
- build in $O(nlogn)$.
- Query in $O(logn)$.
- don't forget to `intit()` after taking $n$ as input.
- don't forget to `start()` before quering
```cpp
vector<vector<ll>>anc, graph, mx, mn;
vector<ll>dpth;
map<pair<ll, ll>, ll>cost;
ll n, m, x, y, c, q;
void init() {
	m = ll(ceil(log2(n)));
	anc = vector<vector<ll>>(n + 1, vector<ll>(m + 1));
	mx = vector<vector<ll>>(n + 1, vector<ll>(m + 1));
	mn = vector<vector<ll>>(n + 1, vector<ll>(m + 1, INF));
	graph = vector<vector<ll>>(n + 1);
	dpth = vector<ll>(n + 1);
}
void dfs(ll i, ll p) {
	for (ll& child : graph[i]) {
		if (child == p)
			continue;
		dpth[child] = dpth[i] + 1;
		anc[child][0] = i;
		mn[child][0] = mx[child][0] = cost[{child, i}];
		for (ll j = 1; j < m; j++) {
			anc[child][j] = anc[anc[child][j - 1]][j - 1];
			mx[child][j] = max(mx[child][j - 1], mx[anc[child][j - 1]][j - 1]);
			mn[child][j] = min(mn[child][j - 1], mn[anc[child][j - 1]][j - 1]);
		}
		dfs(child, i);
	}
}
pair<ll,pair<ll,ll>> k_anc(ll u, ll k) {
	ll mxval = 0, mnVal = INF;
	for (ll i = 0; i < m; i++) {
		if (k & (1LL << i)) {
			mxval = max(mxval, mx[u][i]);
			mnVal = min(mnVal, mn[u][i]);
			u = anc[u][i];
		}
	}
	return { u,{mxval,mnVal} };
}
pair<ll,pair<ll,ll>> lca(ll u, ll v) {
	if (dpth[u] < dpth[v])
		swap(u, v);
	ll k = dpth[u] - dpth[v];
	auto kth = k_anc(u, k);
	u = kth.first;
	ll mxVAl = 0;
	ll mnVal = INF;
	if (u == v)
		return kth;
	for (ll i = m - 1; i >= 0; i--) {
		if (anc[v][i] != anc[u][i]) {
			mxVAl = max(mxVAl, mx[u][i]);
			mxVAl = max(mxVAl, mx[v][i]);
			mnVal = min(mnVal, mn[u][i]);
			mnVal = min(mnVal, mn[v][i]);
			u = anc[u][i];
			v = anc[v][i];
		}
	}
	mxVAl = max(mxVAl, mx[u][0]);
	mxVAl = max(mxVAl, mx[v][0]);
	mnVal = min(mnVal, mn[u][0]);
	mnVal = min(mnVal, mn[v][0]);
	mxVAl = max(mxVAl, kth.second.first);
	mnVal = min(mnVal, kth.second.second);
	return { anc[u][0] ,{mxVAl,mnVal} };
}
void start(ll rt = 1) {
	anc[rt][0] = rt;
	dpth[rt] = 0;
	dfs(rt, -1);
}
void solve() {
	cin >> n;
	init();
	for (ll i = 1; i < n; i++) {
		cin >> x >> y >> c;
		graph[x].push_back(y);
		graph[y].push_back(x);
		cost[{x, y}] = cost[{y, x}] = c;
	}
	start();
	cin >> q;
	while (q--)
	{
		cin >> x >> y;
		auto ans = lca(x, y);
		cout << ans.second.second << " " << ans.second.first << endl;
	}
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
