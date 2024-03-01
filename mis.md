```cpp
double ternary_search(double l, double r) {
    double eps = 1e-9;              //set the error limit here
    while (r - l > eps) {
        double m1 = l + (r - l) / 3;
        double m2 = r - (r - l) / 3;
        double f1 = f(m1);      //evaluates the function at m1
        double f2 = f(m2);      //evaluates the function at m2
        if (f1 < f2)
            l = m1;
        else
            r = m2;
    }
    return f(l);                    //return the maximum of f(x) in [l, r]
}
int lis(vector<int> const& a) {
    int n = a.size();
    const int INF = 1e9;
    vector<int> d(n+1, INF);
    d[0] = -INF;

    for (int i = 0; i < n; i++) {
        int l = upper_bound(d.begin(), d.end(), a[i]) - d.begin();
        if (d[l-1] < a[i] && a[i] < d[l])
            d[l] = a[i];
    }

    int ans = 0;
    for (int l = 0; l <= n; l++) {
        if (d[l] < INF)
            ans = l;
    }
    return ans;
}
#define MOD 1000000007
struct mi {
    int v;
    mi() : mi(0) {}
    mi(int _v) : v(_v) {
        if (v >= MOD) v -= MOD;
        if (v < 0) v += MOD;
    }
    mi(ll _v) : mi((int)(_v % MOD)) {}
    mi operator+(const mi &m2) const { return mi(v + m2.v); }
    mi operator-(const mi &m2) const { return mi(v - m2.v); }
    mi operator*(const mi &m2) const { return mi((ll) v * m2.v); }
    mi operator/(const mi &m2) const { return mi((ll) v * m2.inv().v); }
    mi &operator+=(const mi &m2) { return *this = *this + m2; }
    mi &operator-=(const mi &m2) { return *this = *this - m2; }
    mi &operator*=(const mi &m2) { return *this = *this * m2; }
    mi &operator/=(const mi &m2) { return *this = *this / m2; }
    mi pow(ll e) const {
        mi res = 1;
        mi n = *this;
        while (e > 0) {
            if (e & 1) res *= n;
            n *= n;
            e >>= 1;
        }
        return res;
    }
    mi inv() const {
        return pow(MOD - 2);
    }
};
// goldbach to represent even number as the sum of two prime numbers
pair<int, int> go(int a)
{
  for (int i = 2; i <= a / 2; ++i)
  {
    if (isPrime(i))
    {
      if (isPrime(a - i))
      {
        // cout << a << " = " << i << " + " << a - i << endl;
        return {i, a - i};
      }
    }
  }
  return {0, 0};
}

//fib for any number
#include <map>
#include <iostream>
using namespace std;

#define long long long
const long M = 1000000007; // modulo
map<long, long> F;

long f(long n) {
	if (F.count(n)) return F[n];
	long k=n/2;
	if (n%2==0) { // n=2*k
		return F[n] = (f(k)*f(k) + f(k-1)*f(k-1)) % M;
	} else { // n=2*k+1
		return F[n] = (f(k)*f(k+1) + f(k-1)*f(k)) % M;
	}
}

main(){
	long n;
	F[0]=F[1]=1;
	while (cin >> n)
	cout << (n==0 ? 0 : f(n-1)) << endl;
}
// from the range [1, n]
int findXOR(int n)
{
    int mod = n % 4;
 
    // If n is a multiple of 4
    if (mod == 0)
        return n;
 
    // If n % 4 gives remainder 1
    else if (mod == 1)
        return 1;
 
    // If n % 4 gives remainder 2
    else if (mod == 2)
        return n + 1;
 
    // If n % 4 gives remainder 3
    else if (mod == 3)
        return 0;
}
// Function to return the XOR of elements
// from the range [l, r]
int findXOR(int l, int r)
{
    return (findXOR(l - 1) ^ findXOR(r));
}

```