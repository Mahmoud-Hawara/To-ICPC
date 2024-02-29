# rabin karp
```
gets all the occurences of string s int string t in O(N)  where  N is the maximum between the size of s and size of t
```
```cpp
vector<int> rabin_karp(string const& s, string const& t) {
    const int p = 31; 
    const int m = 1e9 + 9;
    int S = s.size(), T = t.size();

    vector<long long> p_pow(max(S, T)); 
    p_pow[0] = 1; 
    for (int i = 1; i < (int)p_pow.size(); i++) 
        p_pow[i] = (p_pow[i-1] * p) % m;

    vector<long long> h(T + 1, 0); 
    for (int i = 0; i < T; i++)
        h[i+1] = (h[i] + (t[i] - 'a' + 1) * p_pow[i]) % m; 
    long long h_s = 0; 
    for (int i = 0; i < S; i++) 
        h_s = (h_s + (s[i] - 'a' + 1) * p_pow[i]) % m; 

    vector<int> occurences;
    for (int i = 0; i + S - 1 < T; i++) { 
        long long cur_h = (h[i+S] + m - h[i]) % m; 
        if (cur_h == h_s * p_pow[i] % m)
            occurences.push_back(i);
    }
    return occurences;
}
```
# prefix function 
```
finds for every index i the largest suffix that is a prefix in the largest string  in O(N)
```
```cpp
vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi;
}
```
# z function
```
find the largest prefix from the current index i that is also a prefix to the whole string in O(N)
```
```cpp
vector<int> z_function(string s) {
    int n = s.size();
    vector<int> z(n);
    int l = 0, r = 0;
    for(int i = 1; i < n; i++) {
        if(i < r) {
            z[i] = min(r - i, z[i - l]);
        }
        while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if(i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}
```
#aho-corasick
```
The Aho-Corasick algorithm allows us to quickly search for multiple patterns in a text. The set of pattern strings is also called a dictionary. We will denote the total length of its constituent strings by  
$m$  and the size of the alphabet by  
$k$ . The algorithm constructs a finite state automaton based on a trie in  
$O(m k)$  time and then uses it to process the text.
finds fail links in trie using prefix function.
```
``` cpp
const int N = 2e5 + 5;
const ll MOD = 1e9 + 7, MAX = 1e18;
ll n, m = 0, k;
void fastio()
{
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
}
vector<int> g[N];
 
const int K = 26;
struct ACA
{
  vector<int> leafIndices[N];
  int id = 0, pi[N] = {}, trie[N][K] = {};
 
  void insert(int idx, string &s)
  {
    int cur = 0;
    for (auto &i : s)
    {
      if (!trie[cur][i - 'a'])
        trie[cur][i - 'a'] = ++id;
 
      cur = trie[cur][i - 'a'];
    }
 
    leafIndices[cur].push_back(idx);
  }
 
  void build()
  {
    queue<int> q;
 
    for (int i = 0; i < K; i++)
    {
      if (trie[0][i])
        q.push(trie[0][i]);
    }
 
    int cur;
    while (q.size())
    {
      cur = q.front();
 
      q.pop();
 
      for (int i = 0; i < K; i++)
      {
        if (!trie[cur][i])
          continue;
 
        int fail = pi[cur];
        while (fail && !trie[fail][i])
          fail = pi[fail];
 
        fail = trie[fail][i];
        pi[trie[cur][i]] = fail;
        leafIndices[trie[cur][i]].insert(leafIndices[trie[cur][i]].end(),
                                         leafIndices[fail].begin(),
                                         leafIndices[fail].end());
 
        q.push(trie[cur][i]);
      }
    }
  }
 
  int next(int u, char c)
  {
    int cur = u;
    while (cur && !trie[cur][c - 'a'])
      cur = pi[cur];
 
    return trie[u][c - 'a'] = trie[cur][c - 'a'];
  }
 
  void search(string &s, vector<string> v)
  {
    int cur = 0;
    for (int i = 0; i < s.size(); i++)
    {
      cur = next(cur, s[i]);
 
      // String matches
      for (auto &j : leafIndices[cur])
        g[i - v[j].size() + 1].push_back(v[j].size());
    }
  }
};
```
##example usage 
```cpp
ll dp[N];
ll solve(int i)
{
  if (i >= n)
    return 1;
  ll &ret = dp[i];
  if (~ret)
    return ret;
  ret = 0;
  for (auto x : g[i])
  {
    ret = (ret + solve(i + x)) % MOD;
  }
  return ret;
}
void fn()
{
  cin >> n;
 
  vector<string> v(n);
  for (int i = 0; i < n; i++)
    cin >> v[i];
 
  sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
 
  ACA aca;
 
  for (int i = 0; i < v.size(); i++)
    aca.insert(i, v[i]);
  aca.build();
  string s;
  cin >> s;
  aca.search(s, v);
  n = s.size();
  memset(dp, -1, sizeof dp);
  ll ans = solve(0);
  // cout << "ss\n";
  cout << ans << '\n';
}
```
# suffix array
```
finds arrays c , p and lcp in o(n log(n)) for a string 
```
```cpp
// main suffix code
#include<bits/stdc++.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include<set>
#include<vector>
#include<map>
#include<list>
#include<deque>
#define ll long long

using namespace std;


const int N = 3e5+5;
const ll min_num=999999999999999;
void radix_sort( vector<pair<pair<int,int>,int>> &a)
{
    int n=a.size();
  {


        vector<int>cnt(n);
        for(auto x:a)cnt[x.first.second]++;

        vector<pair<pair<int,int>,int>> a_new(n);
        vector<int>pos(n);
        pos[0]=0;
        for(int i=1;i<n;i++)pos[i]=pos[i-1]+cnt[i-1];
        for(auto x:a)
        {
            a_new[pos[x.first.second]]=x;
            pos[x.first.second]++;
        }

  a=a_new;
  }
  {


        vector<int>cnt(n);
        for(auto x:a)cnt[x.first.first]++;

        vector<pair<pair<int,int>,int>> a_new(n);
        vector<int>pos(n);
        pos[0]=0;
        for(int i=1;i<n;i++)pos[i]=pos[i-1]+cnt[i-1];
        for(auto x:a)
        {
            a_new[pos[x.first.first]]=x;
            pos[x.first.first]++;
        }

    a=a_new;
  }

}
int compare(int i, int j, int l) {
    int k=lg[l];
    pair<int, int> a = {c[k][i], c[k][(i+l-(1 << k))%n]};
    pair<int, int> b = {c[k][j], c[k][(j+l-(1 << k))%n]};
    return a == b ? 0 : a < b ? -1 : 1;
}
int lg[N+1];
void pre()
{
  lg[1] = 0;
  for (int i = 2; i <= N; i++)
    lg[i] = lg[i/2] + 1;

}
bool cmp(pair<int,int>a,pair<int,int>b)
{
  int x=compare(a.first,b.first,min(a.second-a.first+1,b.second-b.first+1));
  if(x==-1)return 1;
  if(x==1)return 0;
  if(a.second-a.first+1<b.second-b.first+1)return 1;
  if(a.second-a.first+1>b.second-b.first+1)return 0;
  return a<b;
}
void count_sort(vector<int>&p,vector<int>&c)
{
     int n=p.size();

        vector<int>cnt(n);
        for(auto x:c)cnt[x]++;

        vector<int> p_new(n);
        vector<int>pos(n);
        pos[0]=0;
        for(int i=1;i<n;i++)pos[i]=pos[i-1]+cnt[i-1];
        for(auto x:p)
        {
            p_new[pos[c[x]]]=x;
            pos[c[x]]++;
        }

  p=p_new;

}
/*
c.resize( 22 , vector<int> (n, 0));
    vector<int>p(n);
    //k=0
    {
        vector<pair<char,int>>a(n);
        for(int i=0;i<n;i++)a[i]={s[i],i};
        sort(a.begin(),a.end());
        for(int i=0;i<n;i++)p[i]=a[i].second;
        c[0][p[0]]=0;
        for(int i=1;i<n;i++)
        {
            if(a[i].first==a[i-1].first)c[0][p[i]]=c[0][p[i-1]];
            else c[0][p[i]]=c[0][p[i-1]]+1;
        }
    }

    int k=0;
    while((1<<k) <n)
    {

        for(int i=0;i<n;i++)p[i]=(p[i]-(1<<k)+n)%n;
        count_sort(p,c[k]);
        c[k+1][p[0]]=0;

        for(int i=1;i<n;i++)
            {
                pair<int,int>now={c[k][p[i]],c[k][(p[i]+(1<<k))%n]};
                pair<int,int>prev={c[k][p[i-1]],c[k][(p[i-1]+(1<<k))%n]};
                if(now==prev)c[k+1][p[i]]=c[k+1][p[i-1]];
                else c[k+1][p[i]]=c[k+1][p[i-1]]+1;
            }
            // c=c[k+1];
            k++;

    }
*/
int main() {
    vector<string>vs;
   string s;
   cin>>s;
   s+="$";
   int n=s.size();
   vector<int>p(n),c(n);
   //k=0
   {
       vector<pair<char,int>>a(n);
       for(int i=0;i<n;i++)a[i]={s[i],i};
       sort(a.begin(),a.end());
       for(int i=0;i<n;i++)p[i]=a[i].second;
       c[p[0]]=0;
       for(int i=1;i<n;i++)
       {
           if(a[i].first==a[i-1].first)c[p[i]]=c[p[i-1]];
           else c[p[i]]=c[p[i-1]]+1;
       }
   }

   int k=0;
   while((1<<k) <n)
   {

        for(int i=0;i<n;i++)p[i]=(p[i]-(1<<k)+n)%n;
        count_sort(p,c);
        vector<int>c_new(n);
        c_new[p[0]]=0;

        for(int i=1;i<n;i++)
            {
                pair<int,int>now={c[p[i]],c[(p[i]+(1<<k))%n]};
                pair<int,int>prev={c[p[i-1]],c[(p[i-1]+(1<<k))%n]};
                if(now==prev)c_new[p[i]]=c_new[p[i-1]];
                else c_new[p[i]]=c_new[p[i-1]]+1;
            }
            c=c_new;
            k++;

   }



   vector<int>lcp(n);
    k=0;
   for(int i=0;i<n-1;i++)
   {
       int pi=c[i],j=p[pi-1];
       while(s[i+k]==s[j+k])k++;
       lcp[pi]=k;
       k=max(k-1,0);
   }
    ll sum=0;
   for(int i=0;i<n;i++){
    sum+=n-p[i]-lcp[i]-1;
   }
   cout<<sum<<endl;
   
/*
   int t;
   cin>>t;
   while(t--)
   {
      string s2;
       cin>>s2;
       int l=0,r=n-1,mid,ans=0,st=0,sz=s2.size();
       while(l<=r)
       {
           mid=(l+r)/2;
           //cout<<mid<<endl;
           int ok=0;
           {
               for(int i=0;i<sz;i++)
               {
                   if(s2[i]<s[p[mid]+i])
                   {
                       ok=-1;
                       break;
                   }
                   else if(s2[i]>s[p[mid]+i])
                   {
                       ok=1;
                       break;
                   }
               }
           }
        if(ok<0){
                    r=mid-1;
           }
           else
           {
                if(ok==0)ans=mid;
               l=mid+1;
           }
       }
       st=ans;
       l=0,r=n-1;
       while(l<=r)
       {
           mid=(l+r)/2;
           //cout<<l<<" "<<mid<<" "<<r<<endl;

           int ok=0;
           {
               for(int i=0;i<sz;i++)
               {
                   if(s2[i]<s[p[mid]+i])
                   {
                       ok=-1;
                       break;
                   }
                   else if(s2[i]>s[p[mid]+i])
                   {
                       ok=1;
                       break;
                   }
               }
           }

        if(ok<=0){
                if(ok==0)ans=mid;
                    r=mid-1;
           }
           else
           {
               l=mid+1;
           }
       }
       if(ans==0)cout<<0<<endl;
       else cout<<st-ans+1<<endl;

   }
*/
    return 0;
}

```
#manacher
```
Given string  $s$  with length $n$ . Find all the pairs (i, j)  such that substring s[i....j]  is a palindrome.
```
```cpp

vector<int> manacher_odd(string s) {
    int n = s.size();
    s = "$" + s + "^";
    vector<int> p(n + 2);
    int l = 1, r = 1;
    for(int i = 1; i <= n; i++) {
        p[i] = max(0, min(r - i, p[l + (r - i)]));
        while(s[i - p[i]] == s[i + p[i]]) {
            p[i]++;
        }
        if(i + p[i] > r) {
            l = i - p[i], r = i + p[i];
        }
    }
    return vector<int>(begin(p) + 1, end(p) - 1);
}
vector<int> manacher(string s) {
    string t;
    for(auto c: s) {
        t += string("#") + c;
    }
    auto res = manacher_odd(t + "#");
    return vector<int>(begin(res) + 1, end(res) - 1);
}
```
##some primes under 100
```
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
```
##string stream example
```cpp
    cin >> m;
    cin.ignore();
    while(m--) {
        int x;
        getline(cin, str);
        stringstream ss(str);
        vector<int>v;
        while (ss >> x)v.push_back(x);
    }
```
