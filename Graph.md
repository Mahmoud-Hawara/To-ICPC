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
#dijkestra
```
time: O(E*log(V))
```
```cpp
vector<pair<ll, ll>>graph[N];
ll dist[N];
ll parent[N];
void dijkstra(ll start)
{

    for (int i = 0; i <= n ; i++)
        dist[i] = MAX,parent[i]=0;

    dist[start] = 0;
    priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, greater<pair<ll, ll>>>pq;
    pq.push({0,start});
    while(!pq.empty())
    {
        auto top=pq.top();
        pq.pop();
        ll len=top.first,v=top.second;
        if(len>dist[v])continue;
        for(auto x:graph[v])
        {
            ll to=x.first,l_to=x.second;

            if(dist[to]>len+l_to)
            {
                dist[to]=len+l_to;
                parent[to]=v;
                 pq.push({dist[to],to});
            }
        }
    }
}

```
# floyd

get the distance between every pair of vertexes in $O(n^3)$ time 

```cpp
ll d[N][N];
void floyd()
{
     for (int k = 1; k <= n; ++k) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                 if (d[i][k] < MAX && d[k][j] < MAX)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);

            }
        }
    }
}

```
# MST
form minimum spanning tree in $O(n log(n))$
```cpp
int parent[N],siz[N],sum[N];
void make_set(int v) {
    parent[v] = v;
    siz[v] = 1;
}void init() {
    for (int i = 1; i < N; i++) {
        make_set(i);
    }
    return;
}
int fSet(int v) {
    if (v == parent[v])
        return v;
    return parent[v] = fSet(parent[v]);
}
void uSets(int a, int b) {
    a = fSet(a);
    b = fSet(b);
     if (a != b) {
        if (siz[a] < siz[b])
            swap(a, b);
            parent[b] = a;
	    siz[a] += siz[b];
    }
}
struct Edge {
    int u, v, weight;
    bool operator<(Edge const& other) {
        return weight < other.weight;
    }
};
vector<Edge> edges;
//inside main
init();
int cost = 0;
vector<Edge> result;
sort(edges.begin(), edges.end());

for (Edge e : edges) {
    if (fSet(e.u) != fSet(e.v)) {
        cost += e.weight;
        result.push_back(e);
        uSets(e.u, e.v);
    }
}
```
# euler circuit
This is an algorithm to find an Eulerian circuit in a connected graph in which every vertex has even degree.
1. Choose any vertex v and push it onto a stack. Initially all edges are unmarked.
2. While the stack is nonempty, look at the top vertex, u, on the stack. If u has an unmarked incident edge, say, to a vertex w, then push w onto the stack and mark the edge uw. On the other hand, if u has no unmarked incident edge, then pop u off the stack and print it.                
                 When the stack is empty, you will have printed a sequence of vertices that correspond to an Eulerian circuit

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

int value(char ch){
	if(ch>='a' && ch<='z')	return ch-'a';
	if(ch>='A' && ch<='Z')	return ch-'A'+26;
	return ch-'0'+52;
}

int n, indeg[3900], outdeg[3900];
vector<int> undirected[3900], adj[3900];
bool vis[3900];


void dfs1(int pos)
{
	for (int i = 0; i < undirected[pos].size(); ++i)
	{
		if(!vis[undirected[pos][i]])
		{
			vis[undirected[pos][i]] = true;
			dfs1(undirected[pos][i]);
		}
	}
}

bool is_connected()
{
	for (int i = 0; i < 3900; ++i)
	{
		if(!undirected[i].empty())
		{
			vis[i]=true;
			dfs1(i);
			for (int j = 0; j < 3900; ++j)
			{
				if(!undirected[j].empty() && !vis[j])
					return false;
			}
			return true;
		}
	}
  return 0;
}


int temp_path[400002], final_path[400005], tsz, fsz;

void euler_dfs(int v){

	temp_path[tsz++] = v;
	while(!adj[v].empty()){
		int vv = adj[v].back();
		adj[v].pop_back();
		euler_dfs(vv);
	}
	final_path[fsz++] = temp_path[--tsz];
}


char schar(int val){
	int t = val % 62;
	if(t < 26)	return t+'a';
	t -= 26;
	if(t < 26)	return t+'A';
	t -= 26;
	return t+'0';
}


char fchar(int val){
	return schar(val/62);
}


int main()
{
    //Hierholzer algorithm to find eulerian path
    scanf("%d", &n);
    char str[5];
    int src = -1;
    while(n--){
        scanf("%s", str);
        int a = value(str[0])*62+value(str[1]);
        int b = value(str[1])*62+value(str[2]);
        undirected[a].push_back(b);
        undirected[b].push_back(a);
        outdeg[a]++;	indeg[b]++;
        adj[a].push_back(b);
        src = a;
    }
    int cnt1 = 0, cnt2 = 0;
    for(int i=0;i<62*62;i++){

        if(indeg[i] == 0 && outdeg[i] == 0)	continue;

        if(indeg[i] > outdeg[i]+1 || outdeg[i] > indeg[i]+1){
          printf("NO");
          return 0;
        }

        if(indeg[i] > outdeg[i])	cnt1++;
        if(indeg[i] < outdeg[i]){
          cnt2++;
          src = i;
        }
    }

    if((cnt1==cnt2 && cnt1==0) || (cnt1==cnt2 && cnt1==1)){

        if(!is_connected()){
          printf("NO");
          return 0;
        }

        printf("YES\n");
        euler_dfs(src);

        reverse(final_path, final_path+fsz);
        printf("%c", fchar(final_path[0]));
        for(int i=0;i<fsz;i++)
          printf("%c", schar(final_path[i]));
        return 0;
    }
    
    printf("NO");
    return 0;
}

``` 
# bellman 
see if there negative cycle and calculate minimum path in $O(n^2)$
```cpp
vector<Edge> edges;
const ll INF = 1000000000;
ll xay = 2e10;
bool solve()
{
  vector<ll> d(n, 0);
  vector<ll> p(n, -1);
  ll x;
  for (ll i = 0; i < 1 * n; ++i)
  {
    x = -1;
    for (Edge e : edges)
    {
      if (d[e.a] + e.cost < d[e.b])
      {
        d[e.b] = d[e.a] + e.cost;
        p[e.b] = e.a;
        x = e.b;
        xay = min(xay, d[e.b]);
      }
    }
  }
  if (x == -1)
  {
    return 0;
  }
  else
  {
    return 1;
  }
}

```
# maxflow
find the max flow in graph in $O(v* E^2)$
```cpp
vector<int> graph[3005];
ll capacity[3005][3005];
int src = 1, sink = 2;

ll bfs(int s, int t, vector<int> &parent)
{
  fill(parent.begin(), parent.end(), -1);
  parent[s] = -2;
  queue<pair<int, ll>> q;
  q.push({s, 2e16});

  while (!q.empty())
  {
    int cur = q.front().first;
    ll flow = q.front().second;
    q.pop();

    for (int next : graph[cur])
    {
      if (parent[next] == -1 && capacity[cur][next])
      {
        parent[next] = cur;
        ll new_flow = min(flow, capacity[cur][next]);
        if (next == t)
          return new_flow;
        q.push({next, new_flow});
      }
    }
  }

  return 0;
}
deque<pair<int, int>> ans;
ll maxflow(int s, int t)
{
  ll flow = 0;
  vector<int> parent(n + 5);
  ll new_flow;
  int i = 1;
  while (new_flow = bfs(s, t, parent))
  {
    flow += new_flow;
    int cur = t;
    int ok = 1;
    while (cur != s)
    {
      int prev = parent[cur];
      capacity[prev][cur] -= new_flow;
      capacity[cur][prev] += new_flow;
      if (capacity[prev][cur] == 0 && ok)
      {
        ok = 0;
        // cout << i << '\n';
        ans.push_back({prev, cur});
      }
      cur = prev;
    }
    i++;
  }

  return flow;
}
int vis[N];
void dfs(int x, int a = 1)
{
  vis[x] = a;
  // cout << x << " s " << a << '\n';
  for (auto i : graph[x])
  {
    if (!vis[i])
    {
      if (capacity[x][i] || a == 2)
        dfs(i, a);
    }
    else if (vis[i] == 1 && a == 2)
    {
      cout << x << " " << i << '\n';
    }
  }
}
void addedge(int a, int b, int c)
{
  capacity[a][b] += x;
  // capacity[b][a] += x; //if the graph is undirected uncomment this
  graph[a].push_back(b);
  graph[b].push_back(a);
}
```
# bridges
find all bridges in the graph in $O(V+E)$
```cpp
vector<pair<int, int>> graph[N];
vector<int> graph2[N];
int vis[N];
vector<int> tin, low;
int timer;
stack<int> stk;
int root[N];
vector<pair<int, int>> pr;
void dfs(int v, int p = -1)
{
  vis[v] = true;
  stk.push(v);
  tin[v] = low[v] = timer++;
  for (pair<int, int> to : graph[v])
  {
    if (to.second == p)
      continue;
    if (vis[to.first])
    {
      low[v] = min(low[v], tin[to.first]);
    }
    else
    {
      dfs(to.first, to.second);
      low[v] = min(low[v], low[to.first]);
      if (low[to.first] > tin[v])
        pr.push_back({to.first, v});
    }
  }
  if (low[v] == tin[v])
  {
    int cnt = 1;
    while (stk.top() != v)
    {
      root[stk.top()] = v;
      stk.pop();
      cnt++;
    }
    stk.pop();
  }
}
 
void find_bridges()
{
  timer = 0;
  tin.assign(n, -1);
  pr.clear();
  low.assign(n, -1);
  for (int i = 0; i < n; ++i)
  {
    if (!vis[i])
      dfs(i);
  }
}
```
# bridges online 
```cpp
vector<int> par, dsu_2ecc, dsu_cc, dsu_cc_size;
int bridges;
int lca_iteration;
vector<int> last_visit;

void init(int n) {
    par.resize(n);
    dsu_2ecc.resize(n);
    dsu_cc.resize(n);
    dsu_cc_size.resize(n);
    lca_iteration = 0;
    last_visit.assign(n, 0);
    for (int i=0; i<n; ++i) {
        dsu_2ecc[i] = i;
        dsu_cc[i] = i;
        dsu_cc_size[i] = 1;
        par[i] = -1;
    }
    bridges = 0;
}

int find_2ecc(int v) {
    if (v == -1)
        return -1;
    return dsu_2ecc[v] == v ? v : dsu_2ecc[v] = find_2ecc(dsu_2ecc[v]);
}

int find_cc(int v) {
    v = find_2ecc(v);
    return dsu_cc[v] == v ? v : dsu_cc[v] = find_cc(dsu_cc[v]);
}

void make_root(int v) {
    v = find_2ecc(v);
    int root = v;
    int child = -1;
    while (v != -1) {
        int p = find_2ecc(par[v]);
        par[v] = child;
        dsu_cc[v] = root;
        child = v;
        v = p;
    }
    dsu_cc_size[root] = dsu_cc_size[child];
}

void merge_path (int a, int b) {
    ++lca_iteration;
    vector<int> path_a, path_b;
    int lca = -1;
    while (lca == -1) {
        if (a != -1) {
            a = find_2ecc(a);
            path_a.push_back(a);
            if (last_visit[a] == lca_iteration){
                lca = a;
                break;
                }
            last_visit[a] = lca_iteration;
            a = par[a];
        }
        if (b != -1) {
            b = find_2ecc(b);
            path_b.push_back(b);
            if (last_visit[b] == lca_iteration){
                lca = b;
                break;
                }
            last_visit[b] = lca_iteration;
            b = par[b];
        }

    }

    for (int v : path_a) {
        dsu_2ecc[v] = lca;
        if (v == lca)
            break;
        --bridges;
    }
    for (int v : path_b) {
        dsu_2ecc[v] = lca;
        if (v == lca)
            break;
        --bridges;
    }
}

void add_edge(int a, int b) {
    a = find_2ecc(a);
    b = find_2ecc(b);
    if (a == b)
        return;

    int ca = find_cc(a);
    int cb = find_cc(b);

    if (ca != cb) {
        ++bridges;
        if (dsu_cc_size[ca] > dsu_cc_size[cb]) {
            swap(a, b);
            swap(ca, cb);
        }
        make_root(a);
        par[a] = dsu_cc[a] = b;
        dsu_cc_size[cb] += dsu_cc_size[a];
    } else {
        merge_path(a, b);
    }
}
```
# articulation
```cpp
int n; // number of nodes
vector<vector<int>> adj; // adjacency list of graph

vector<bool> visited;
vector<int> tin, low;
int timer;

void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    int children=0;
    for (int to : adj[v]) {
        if (to == p) continue;
        if (visited[to]) {
            low[v] = min(low[v], tin[to]);
        } else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] >= tin[v] && p!=-1)
                IS_CUTPOINT(v);
            ++children;
        }
    }
    if(p == -1 && children > 1)
        IS_CUTPOINT(v);
}

void find_cutpoints() {
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs (i);
    }
}
```
# hungarian
```cpp
double arr[105][105];
vector<int> hungarian()
{
  m = n;
  vector<double> u(n + 1), v(m + 1);
  vector<int> p(m + 1), way(m + 1);
  for (int i = 1; i <= n; ++i)
  {
    p[0] = i;
    int j0 = 0;
    vector<double> minv(m + 1, INF);
    vector<bool> used(m + 1, false);
    do
    {
      used[j0] = true;
      int i0 = p[j0], j1;
      double delta = INF;
      for (int j = 1; j <= m; ++j)
        if (!used[j])
        {
          double cur = arr[i0][j] - u[i0] - v[j];
          if (cur < minv[j])
            minv[j] = cur, way[j] = j0;
          if (minv[j] < delta)
            delta = minv[j], j1 = j;
        }
      for (int j = 0; j <= m; ++j)
        if (used[j])
          u[p[j]] += delta, v[j] -= delta;
        else
          minv[j] -= delta;
      j0 = j1;
    } while (p[j0] != 0);
    do
    {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  vector<int> ans(n + 1);
  for (int j = 1; j <= m; ++j)
    ans[j] = p[j];
  return ans;
}
```
# SCC
```cpp
vector<vector<int>> adj, adj_rev;
vector<bool> used;
vector<int> order, component;

void dfs1(int v) {
    used[v] = true;

    for (auto u : adj[v])
        if (!used[u])
            dfs1(u);

    order.push_back(v);
}

void dfs2(int v) {
    used[v] = true;
    component.push_back(v);

    for (auto u : adj_rev[v])
        if (!used[u])
            dfs2(u);
}

int main() {
    int n;
    // ... read n ...

    for (;;) {
        int a, b;
        // ... read next directed edge (a,b) ...
        adj[a].push_back(b);
        adj_rev[b].push_back(a);
    }

    used.assign(n, false);

    for (int i = 0; i < n; i++)
        if (!used[i])
            dfs1(i);

    used.assign(n, false);
    reverse(order.begin(), order.end());

    for (auto v : order)
        if (!used[v]) {
            dfs2 (v);

            // ... processing next component ...

            component.clear();
        }
}
vector<int> roots(n, 0);
vector<int> root_nodes;
vector<vector<int>> adj_scc(n);

for (auto v : order)
    if (!used[v]) {
        dfs2(v);

        int root = component.front();
        for (auto u : component) roots[u] = root;
        root_nodes.push_back(root);

        component.clear();
    }


for (int v = 0; v < n; v++)
    for (auto u : adj[v]) {
        int root_v = roots[v],
            root_u = roots[u];

        if (root_u != root_v)
            adj_scc[root_v].push_back(root_u);
    }
```
# 2-sat
```cpp
vector<int> adj[N], adj_t[N];
vector<bool> used;
vector<int> order, comp;
vector<bool> assignment;
 
void dfs1(int v)
{
  used[v] = true;
  for (int u : adj[v])
  {
    if (!used[u])
      dfs1(u);
  }
  order.push_back(v);
}
 
void dfs2(int v, int cl)
{
  comp[v] = cl;
  for (int u : adj_t[v])
  {
    if (comp[u] == -1)
      dfs2(u, cl);
  }
}
 
bool solve_2SAT()
{
  order.clear();
  used.assign(n, false);
  for (int i = 0; i < n; ++i)
  {
    if (!used[i])
      dfs1(i);
  }
 
  comp.assign(n, -1);
  for (int i = 0, j = 0; i < n; ++i)
  {
    int v = order[n - i - 1];
    if (comp[v] == -1)
      dfs2(v, j++);
  }
 
  assignment.assign(n / 2, false);
  for (int i = 0; i < n; i += 2)
  {
    if (comp[i] == comp[i + 1])
      return false;
    assignment[i / 2] = comp[i] > comp[i + 1];
  }
  return true;
}
 
void add_disjunction(int a, bool na, int b, bool nb)
{
  // na and nb signify whether a and b are to be negated
  a = 2 * a ^ na;
  b = 2 * b ^ nb;
  int neg_a = a ^ 1;
  int neg_b = b ^ 1;
  adj[neg_a].push_back(b);
  adj[neg_b].push_back(a);
  // cout << neg_a << " " << b << " s " << neg_b << " " << a << '\n';
  adj_t[b].push_back(neg_a);
  adj_t[a].push_back(neg_b);
}
vector<int> vv[N];
ll arr[N];
void fn()
{
  cin >> m >> n;
  for (int i = 1; i <= m; i++)
  {
    cin >> arr[i];
  }
  for (int i = 1; i <= n; i++)
  {
    int sz;
    cin >> sz;
    for (int j = 1; j <= sz; j++)
    {
      int a;
      cin >> a;
      vv[a].push_back(i - 1);
    }
  }
  for (int i = 1; i <= m; i++)
  {
    // cout << arr[i] << " " << vv[i].size() << " " << vv[i][0] << " " << vv[i][1] << '\n';
    if (arr[i])
    {
      add_disjunction(vv[i][0], 0, vv[i][1], 1);
      add_disjunction(vv[i][0], 1, vv[i][1], 0);
    }
    else
    {
      add_disjunction(vv[i][0], 1, vv[i][1], 1);
      add_disjunction(vv[i][0], 0, vv[i][1], 0);
    }
  }
  n *= 2;
  bool ok = solve_2SAT();
  if (ok)
  {
    cout << "YES\n";
    // for (auto x : assignment)
    // {
    //   cout << x << " ";
    // }
  }
  else
    cout << "NO\n";
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
## find all paths in tree of at least length k
```cpp
vector<int> Adj[N];

int k;
long long ans = 0;

vector<int> dfs(int u, int p)
{
  vector<int> V;
  for (int v : Adj[u])
  {
    if (v == p)
      continue;
    vector<int> X = dfs(v, u);

    // Small to Large Merging
    if (V.size() < X.size())
      swap(V, X);

    // number of paths of length >= k passing through u
    for (int i = 0; i < X.size(); i++)
    {
      // number of nodes at depth d = cnt
      int cnt = X[i];
      if (i)
        cnt -= X[i - 1];
      int d = X.size() - i;
      // search for nodes with depth atleast k - d
      if (d >= k)
        ans += 1ll * cnt * V.back();
      else
      {
        if (k - d > V.size())
          break;
        ans += 1ll * cnt * V[V.size() - (k - d)];
      }
    }

    // Merge X with V
    for (int i = 0; i < X.size(); i++)
    {
      int depth = X.size() - i;
      int cnt = X[i];
      V[V.size() - depth] += cnt;
    }
  }

  // number of paths starting from u
  if (V.size() >= k)
    ans += V[V.size() - k];

  if (V.empty())
    V.push_back(1);
  else
    V.push_back(V.back() + 1);

  // V = Prefix_sum(number of vertices at depth in reversed order)
  return V;
}

``` 