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
```cpp
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#define ll long long
#define ld long double
using namespace std;
using namespace __gnu_pbds;
typedef pair<int, int> node;
typedef tree<node, null_type, less<node>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
const int N = 5e5 + 5;
const ll MOD = 1e9 + 7, MAX = 1e18;
const double ep = 1e-6, pi = atan(1.0) * 4;
long long inv(long long a, long long b = MOD)
{
    return 1 < a ? b - inv(b % a, a) * b / a : 1;
}
int dx[8] = {1, 0, -1, 0, -1, 1, -1, 1};
int dy[8] = {0, 1, 0, -1, -1, 1, 1, -1};
```
# FFT
All possible sums¶
We are given two arrays  
$a[]$  and  
$b[]$ . We have to find all possible sums  
$a[i] + b[j]$ , and for each sum count how often it appears.

For example for  
$a = [1,~ 2,~ 3]$  and  
$b = [2,~ 4]$  we get: then sum  
$3$  can be obtained in  
$1$  way, the sum  
$4$  also in  
$1$  way,  
$5$  in  
$2$ ,  
$6$  in  
$1$ ,  
$7$  in  
$1$ .

We construct for the arrays  
$a$  and  
$b$  two polynomials  
$A$  and  
$B$ . The numbers of the array will act as the exponents in the polynomial ( 
$a[i] \Rightarrow x^{a[i]}$ ); and the coefficients of this term will be how often the number appears in the array.

Then, by multiplying these two polynomials in  
$O(n \log n)$  time, we get a polynomial  
$C$ , where the exponents will tell us which sums can be obtained, and the coefficients tell us how often. To demonstrate this on the example:

 
$$(1 x^1 + 1 x^2 + 1 x^3) (1 x^2 + 1 x^4) = 1 x^3 + 1 x^4 + 2 x^5 + 1 x^6 + 1 x^7$$ 
All possible scalar products¶
We are given two arrays  
$a[]$  and  
$b[]$  of length  
$n$ . We have to compute the products of  
$a$  with every cyclic shift of  
$b$ .

We generate two new arrays of size  
$2n$ : We reverse  
$a$  and append  
$n$  zeros to it. And we just append  
$b$  to itself. When we multiply these two arrays as polynomials, and look at the coefficients  
$c[n-1],~ c[n],~ \dots,~ c[2n-2]$  of the product  
$c$ , we get:

  
 
 
$$c[k] = \sum_{i+j=k} a[i] b[j]$$ 
And since all the elements  
$a[i] = 0$  for  
$i \ge n$ :

 
 
 
$$c[k] = \sum_{i=0}^{n-1} a[i] b[k-i]$$ 
It is easy to see that this sum is just the scalar product of the vector  
$a$  with the  
$(k - (n - 1))$ -th cyclic left shift of  
$b$ . Thus these coefficients are the answer to the problem, and we were still able to obtain it in  
$O(n \log n)$  time. Note here that  
$c[2n-1]$  also gives us the  
$n$ -th cyclic shift but that is the same as the  
$0$ -th cyclic shift so we don't need to consider that separately into our answer.

Two stripes¶
We are given two Boolean stripes (cyclic arrays of values  
$0$  and  
$1$ )  
$a$  and  
$b$ . We want to find all ways to attach the first stripe to the second one, such that at no position we have a  
$1$  of the first stripe next to a  
$1$  of the second stripe.

The problem doesn't actually differ much from the previous problem. Attaching two stripes just means that we perform a cyclic shift on the second array, and we can attach the two stripes, if scalar product of the two arrays is  
$0$ .

String matching¶
We are given two strings, a text  
$T$  and a pattern  
$P$ , consisting of lowercase letters. We have to compute all the occurrences of the pattern in the text.

We create a polynomial for each string ( 
$T[i]$  and  
$P[I]$  are numbers between  
$0$  and  
$25$  corresponding to the  
$26$  letters of the alphabet):

 
$$A(x) = a_0 x^0 + a_1 x^1 + \dots + a_{n-1} x^{n-1}, \quad n = |T|$$ 
with

 
 
 
$$a_i = \cos(\alpha_i) + i \sin(\alpha_i), \quad \alpha_i = \frac{2 \pi T[i]}{26}.$$ 
And

 
$$B(x) = b_0 x^0 + b_1 x^1 + \dots + b_{m-1} x^{m-1}, \quad m = |P|$$ 
with

 
 
 
$$b_i = \cos(\beta_i) - i \sin(\beta_i), \quad \beta_i = \frac{2 \pi P[m-i-1]}{26}.$$ 
Notice that with the expression  
$P[m-i-1]$  explicitly reverses the pattern.

The  
$(m-1+i)$ th coefficients of the product of the two polynomials  
$C(x) = A(x) \cdot B(x)$  will tell us, if the pattern appears in the text at position  
$i$ .

 
 
 
 
 
$$c_{m-1+i} = \sum_{j = 0}^{m-1} a_{i+j} \cdot b_{m-1-j} = \sum_{j=0}^{m-1} \left(\cos(\alpha_{i+j}) + i \sin(\alpha_{i+j})\right) \cdot \left(\cos(\beta_j) - i \sin(\beta_j)\right)$$ 
with  
 
 
$\alpha_{i+j} = \frac{2 \pi T[i+j]}{26}$  and  
 
 
$\beta_j = \frac{2 \pi P[j]}{26}$ 

If there is a match, than  
$T[i+j] = P[j]$ , and therefore  
$\alpha_{i+j} = \beta_j$ . This gives (using the Pythagorean trigonometric identity):

  
 
 
 
 
 
 
 
 
$$\begin{align} c_{m-1+i} &= \sum_{j = 0}^{m-1} \left(\cos(\alpha_{i+j}) + i \sin(\alpha_{i+j})\right) \cdot \left(\cos(\alpha_{i+j}) - i \sin(\alpha_{i+j})\right) \\ &= \sum_{j = 0}^{m-1} \cos(\alpha_{i+j})^2 + \sin(\alpha_{i+j})^2 = \sum_{j = 0}^{m-1} 1 = m \end{align}$$ 
If there isn't a match, then at least a character is different, which leads that one of the products  
$a_{i+1} \cdot b_{m-1-j}$  is not equal to  
$1$ , which leads to the coefficient  
$c_{m-1+i} \ne m$ .

String matching with wildcards¶
This is an extension of the previous problem. This time we allow that the pattern contains the wildcard character  
$\*$ , which can match every possible letter. E.g. the pattern  
$a*c$  appears in the text  
$abccaacc$  at exactly three positions, at index  
$0$ , index  
$4$  and index  
$5$ .

We create the exact same polynomials, except that we set  
$b_i = 0$  if  
$P[m-i-1] = *$ . If  
$x$  is the number of wildcards in  
$P$ , then we will have a match of  
$P$  in  
$T$  at index  
$i$  if  
$c_{m-1+i} = m - x$ .
