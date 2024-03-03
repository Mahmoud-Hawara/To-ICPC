# Matching
```cpp
// O(n * k)

int n, k; // n is the number of nodes in the first part, k is the number in the second part
vector<vector<int>> g;
vector<int> mt;
vector<bool> used;

bool canMatch(int v)
{
    if (used[v])return false;
    used[v] = true;
    for (int to : g[v])
    {
        if (mt[to] == -1 || canMatch(mt[to]))
        {
            mt[to] = v;
            return true;
        }
    }
    return false;
}

int main()
{
    // ... reading the bipartite graph ...

    mt.assign(k + 1, -1);
    vector<bool> used1(n + 1, false);
    for (int v = 1; v <= n; v++)
    {
        for (int to : g[v])
        {
            if (mt[to] == -1)
            {
                mt[to] = v;
                used1[v] = true;
                break;
            }
        }
    }
    for (int v = 1; v <= n; v++)
    {
        if (used1[v])continue;
        used.assign(n + 1, false);
        canMatch(v);
    }

    for (int i = 1; i <= k; i++)
        if (mt[i] != -1)
        {
            cout << mt[i] << ' ' << i << '\n';
        }
}






#include <bits/stdc++.h>

using namespace std;

#define IO ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
#define ll long long

const int N = 2e5 + 5;
const ll MOD = 1e9 + 7;

int n, m, k; 
vector<int> g[N], mt;
vector<bool> used;

bool canMatch(int v)
{
    if (used[v])return false;
    used[v] = true;
    for (int to : g[v])
    {
        if (mt[to] == -1 || canMatch(mt[to]))
        {
            mt[to] = v;
            return true;
        }
    }
    return false;
}
void solve()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        int k;
        cin >> k;
        for (int j = 1; j <= k; j++)
        {
            int x;
            cin >> x;
            g[i].push_back(n + x);
            g[n + x].push_back(i);
        }
    }
    mt.assign(n + m + 1, -1);
    vector<bool> used1(n + 1, false);
    for (int v = 1; v <= n; v++)
    {
        for (int to : g[v])
        {
            if (mt[to] == -1)
            {
                mt[to] = v;
                used1[v] = true;
                break;
            }
        }
    }
    for (int v = 1; v <= n; v++)
    {
        if (used1[v])continue;
        used.assign(n + 1, false);
        canMatch(v);
    }
    int ans = 0;
    for (int i = n + 1; i <= n + m; i++)
        if (mt[i] != -1)
        {
            ans++;
        }
    cout << ans;
    return;
}

int main()
{
    IO 
    int t = 1;
    // cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}
```

## Problem solved with it
```cpp
#include <bits/stdc++.h>

using namespace std;

#define IO ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
#define ll long long

const int N = 2e5 + 5;
const ll MOD = 1e9 + 7;

int n, m, k; 
vector<int> g[N], mt;
vector<bool> used;

bool canMatch(int v)
{
    if (used[v])return false;
    used[v] = true;
    for (int to : g[v])
    {
        if (mt[to] == -1 || canMatch(mt[to]))
        {
            mt[to] = v;
            return true;
        }
    }
    return false;
}
void solve()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        int k;
        cin >> k;
        for (int j = 1; j <= k; j++)
        {
            int x;
            cin >> x;
            g[i].push_back(n + x);
            g[n + x].push_back(i);
        }
    }
    mt.assign(n + m + 1, -1);
    vector<bool> used1(n + 1, false);
    for (int v = 1; v <= n; v++)
    {
        for (int to : g[v])
        {
            if (mt[to] == -1)
            {
                mt[to] = v;
                used1[v] = true;
                break;
            }
        }
    }
    for (int v = 1; v <= n; v++)
    {
        if (used1[v])continue;
        used.assign(n + 1, false);
        canMatch(v);
    }
    int ans = 0;
    for (int i = n + 1; i <= n + m; i++)
        if (mt[i] != -1)
        {
            ans++;
        }
    cout << ans;
    return;
}

int main()
{
    IO 
    int t = 1;
    // cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}
```
