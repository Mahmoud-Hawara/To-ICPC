# extended GCD

- representation of `gcd(a, b)` in form of `a*x + b *y = gcd(a, b)`
- x: the multiplication inverse of a % b
- time: O(log n)

``` c++
int extended_gcd(int a, int b, int &x, int &y)
{
    x = 1, y = 0;
    int x1 = 0, y1 = 1, a1 = a, b1 = b;
    while (b1)
    {
        int q = a1 / b1;
        tie(x, x1) = make_tuple(x1, x - q * x1);
        tie(y, y1) = make_tuple(y1, y - q * y1);
        tie(a1, b1) = make_tuple(b1, a1 - q * b1);
    }
    return a1;
}
int rec_gcd(int a, int b, int &x, int &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    int x1, y1;
    int d = rec_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}
```

# Diophantine equation
- For `a*x + b*y = c`, find any solution where $g$ is $gcd(a, b)$.
- If $c$ is multiple of $gcd(a, b)$, there is a solution where for `a*x + b*y = c`. otherwise, $NO$ solution.
- Time: O(log n)

```c++
bool find_any_solution(int a, int b, int c, int &x0, int &y0, int &g)
{
    g = rec_gcd(abs(a), abs(b), x0, y0);
    if (c % g)
    {
        return false;
    }
    x0 *= c / g;
    y0 *= c / g;
    if (a < 0)
        x0 = -x0;
    if (b < 0)
        y0 = -y0;
    return true;
}
```

- the solution for $x+y$ is minimum possible: 
use the information in `shift_solution()` from all possible solutions.
- If a < b , we need to select smallest possible value of  cnt.
- If a > b , we need to select the largest possible value of  cnt .
- If  a = b , all solution will have the same sum x + y.

<h5> OR: just get min sum when shift_solution is called every time</h5>

```c++
int mnsum = 1e9 + 7, xmnsm = 0, ymnsm = 0;
void shift_solution(int &x, int &y, int a, int b, int cnt)
{
    if (mnsum > x + y)
    {
        mnsum = x + y;
        xmnsm = x, ymnsm = y;
    }
    x += cnt * b;
    y -= cnt * a;
    if (mnsum > x + y)
    {
        mnsum = x + y;
        xmnsm = x, ymnsm = y;
    }
    cout << x << " " << y << " " << cnt << endl;
}
```

- <h4>the function gets the number of solutions in a range for x and y</h4>
```c++
int find_all_solutions(int a, int b, int c, int minx, int maxx, int miny, int maxy)
{
    int x, y, g;
    if (!find_any_solution(a, b, c, x, y, g))
        return 0;
    a /= g;
    b /= g;

    int sign_a = a > 0 ? +1 : -1;
    int sign_b = b > 0 ? +1 : -1;

    shift_solution(x, y, a, b, (minx - x) / b);

    if (x < minx)
        shift_solution(x, y, a, b, sign_b);

    if (x > maxx)
        return 0;
    int lx1 = x;

    shift_solution(x, y, a, b, (maxx - x) / b);

    if (x > maxx)
        shift_solution(x, y, a, b, -sign_b);

    int rx1 = x;

    shift_solution(x, y, a, b, -(miny - y) / a);

    if (y < miny)
        shift_solution(x, y, a, b, -sign_a);

    if (y > maxy)
        return 0;
    int lx2 = x;

    shift_solution(x, y, a, b, -(maxy - y) / a);

    if (y > maxy)
        shift_solution(x, y, a, b, sign_a);

    int rx2 = x;

    if (lx2 > rx2)
        swap(lx2, rx2);
    int lx = max(lx1, lx2);
    int rx = min(rx1, rx2);

    if (lx > rx)
        return 0;
    return (rx - lx) / abs(b) + 1;
}
```

# Mobius function

- $m(1) = 1$ 
- $m(n) = 1$ if $n$ is a square free positive integer + Even number of prime factors: $m(2.3.5.7) = 1$
- $m(n) = -1$ if $n$ is a square free positive integer + Odd number of prime factors: $m(2.3.5) = -1$
- $m(n) = 0$ if n is $NOT$ a square free integer: $m(2.3.3.7) = 0$

- `M(n)  = 1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1`
- `M(n) + 1 = 2, 0, 0, 1, 0, 2, 0, 1, 1, 2, 0, 1, 0, 2` 

- Time: $O(sqrt(n))$
```c++
ll mobius(ll n)
{
    ll mebVal = 1;
    for (ll i = 2; i * i <= n; i++)
    {
        if (n % i == 0)
        {
            if (n % (i * i) == 0)
                return 0;
            n /= i;
            mebVal *= -1;
        }
    }
    if (n)
        mebVal *= -1;
    return mebVal;
}
```

#### generate the mobius for elements in range from 1 to n.
- the same technique as seive.

```c++
vector<int> mobius_generator(int n)
{
    vector<int> mobius(n + 1, -1);
    vector<bool> prime(n + 1, 1);

    mobius[1] = 1, prime[1] = 0;

    for (int i = 2; i <= n; i++)
    {
        if (!prime[i])
            continue;
        mobius[i] = -1;
        for (int j = i * i; j <= n; j += i)
        {
            prime[j] = 0;
            mobius[j] = j % (i * i) == 0 ? 0 : -mobius[j];
        }
    }
    return mobius;
}
```

### Mobius uses.
- `kth_square_free(kth, n)`: get the k-th square free number in a range with O(n). 
- `coprime_triples(n)`: count triples $(a, b, c)$ where $a, b, c <= n$ and $gcd(a, b,c) = 1$

```c++
int kth_square_free(int k, int n)
{
    auto mob = mobius_generator(n);
    vector<int> v(n);
    for (int i = 1; i <= n; i++)
        v[i] = mob[i] != 0;
    for (int i = 2; i <= n; i++)
        v[i] += v[i - 1];
    return v[k];
}

ll coprime_triples(ll n)
{
    auto mobius = mobius_generator(n);
    ll sum = 0;
    for (ll i = 2; i <= n; i++)
    {
        sum -= mobius[i] * (n / i) * (n / i) * (n / i);
    }
    return sum;
}
ll square_free_index(ll val, int n)
{
    auto mobius = mobius_generator(n);
    ll idx = val;
    for (ll i = 2; i * i <= val; i++)
        idx += mobius[i] * (val / (i * i));

    return idx;
}
```
# Gaussian Elimination
- Given a system of $n$ linear algebraic equations with $m$ unknowns. You are asked to solve the system: to determine if it has no solution, exactly one solution or infinite number of solutions. And in case it has at least one solution, find any of them.
- Search and reshuffle the pivoting row. This takes $O(n + m)$  when using heuristic mentioned above.
- If the pivot element in the current column is found - then we must add this equation to all other equations, which takes time $O(nm)$ so it will be $O(n^3)$.
```cpp
pair<int, int> prepareFraction(int n, int d)
{
	int div = gcd(n, d);
	n /= div, d /= div;
	if(d < 0)	n *= -1,  d*= -1;
	return make_pair(n, d);
}

/*
0X + 0Y = 0		,   0X +  0Y  = 4  		--> NO_SOLUTIONS
0X + 0Y = 0		,   0X +  0Y  = 0 		--> INFINITE_SOLUTIONS
2X + 3Y = 9		,  -2X + -3Y  = -9		--> INFINITE_SOLUTIONS
1X + 3Y = 1		,   2X +  6Y  = -1		--> NO_SOLUTIONS
1X + 2Y = 6		,   1X +  -4Y = -3		--> X=3/1, Y=3/2

-9X + -9Y = -9	, -7X + -8Y = -8		--> X=0/1, Y=1/1
-6X +  67 = -7	, -1X +  1Y = -6		--> NO_SOLUTIONS
-7X +  2Y =  1	, -7X + -7Y =  6		--> X=(-19)/63 Y=(-5)/9
 9X + -2Y =  6	, -1X + -2Y = -1  	  	-->	X=7/10 Y=3/20
 1X +  1Y =  1	,  0X +  0Y =  0		--> INFINITE_SOLUTIONS
*/
pair<string, vector<int> > solve2Equations( int ax, int ay, int az,
											int bx, int by, int bz)
{
	if( (!ax && !ay && az) || (!bx && !by && bz) )
		return make_pair("NO_SOLUTIONS", vector<int>(4) );

	if( (!ax && !ay && !az) || (!bx && !by && !bz) )
		return make_pair("INFINITE_SOLUTIONS", vector<int>(4) );

    if( ax*by == ay*bx && ax*bz==az*bx && ay*bz==by*az )
      	return make_pair("INFINITE_SOLUTIONS", vector<int>(4) );

    if( ax*by == ay*bx )
      	return make_pair("NO_SOLUTIONS", vector<int>(4) );

    pair<int, int> X = prepareFraction(by*az-ay*bz, by*ax-bx*ay);
    pair<int, int> Y = prepareFraction(bx*az-ax*bz, bx*ay-by*ax);

    int sol[] = {X.first, X.second, Y.first, Y.second };
    return make_pair("SOLVED", vector<int>(sol, sol+4) );
}

const double EPS = (1e-10);

# define isZero(c) 	(fabs(c) < EPS)
enum GaussStatus {	NO_SOLUTION = -10, UNIQUE = 1, INFINITE_SOLUTION = 10	};

bool IsInvalidEqu(int c, vector<double> &row) {
	// 0x+0y = 10 is invalid equ
	for(;c < (int)row.size()-1;++c) if(!isZero(row[c]))
		return false;

	return isZero(row.back()) ? false : true;
}


/*
// E.g. Solve: x + y = 3	x - y = 7
vector<vector<double>> mat(2);
mat[0] = {1,  1, 3};
mat[1] = {1, -1, 7};

int sol = solveLinerEqu(mat);	=> UNIQUE
=> mat[0][2] = 5, mat[1][2] = -2
*/
GaussStatus GaussElim(vector<vector<double>> & mat){
	int n = mat.size(), m = mat[0].size()-1;
	GaussStatus status = UNIQUE;

	for(int r = 0, c = 0; r < n && c < m; ++r, ++c){
		int stable_r = r;	// partial pivoting
		for(int j = r + 1; j < n; ++j)
			if(fabs(mat[j][c]) > fabs(mat[stable_r][c]))
				stable_r = j;

		if(stable_r != r)
			swap(mat[stable_r], mat[r]);
		else if(isZero(mat[r][c])){
			if(IsInvalidEqu(c, mat[r]))
				return NO_SOLUTION;
			--r, status = INFINITE_SOLUTION;	// c-th variable can be anything
			continue;
		}
		for(int j = m; j >= c; --j) // convert diagonal to 1
			mat[r][j] /= mat[r][c];

		// zero all my column, except current row. Optimize for spare matrix
		for(int k = 0; k < n; ++k) if(k != r && !isZero(mat[k][c])) {
			for(int j = m; j >= c; --j)	// Add to kth-row: -mat[k][c] * mat[r] to zero it
				mat[k][j] -= mat[k][c] * mat[r][j];
		}
	}	// watch out from solutions -0.0000000001
	// To compute rank, see how many non zero equations
	return status;
}

ll pow(ll a, ll k, ll M) {
	if (k == 0)		return 1;
	ll r = pow(a, k / 2, M);
	r *= r, r %= M;
	if (k % 2)	r = (r * a) % M;
	return r;
}

bool IsInvalidEqu(int c, vector<ll> &row) {
	// 0x+0y = 10 is invalid equ
	for(;c < (int)row.size()-1;++c) if(row[c] != 0)
		return false;

	return row.back() == 0 ? false : true;
}


/*
Solve system of linear equations % p
x + y = 5  (% 5)
3x + 6y = 1  (% 5)
x = 3 and y = 2 solves this system
Caution: NOT TESTED ON OJ

vector<vector<int >> mat(2);
mat[0] = {1, 1, 5};
mat[1] = {3, 6, 1};

GaussElimModPrime(mat, 5);
*/
GaussStatus GaussElimModPrime(vector<vector<ll>> & mat, int p){
	int n = mat.size(), m = mat[0].size()-1;
	GaussStatus status = UNIQUE;

	// Fix any -ve mode or higher than p
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j <= m; ++j)
			mat[i][j] = (mat[i][j] % p + p) % p;	// mod expensive
	}

	for(int r = 0, c = 0; r < n && c < m; ++r, ++c){
		int stable_r = r;
		for(int j = r; j < n; ++j)
			if(mat[j][c] != 0) {
				stable_r = j;
				break;	// any value that can be mod inversed
			}

		if(stable_r != r)
			swap(mat[stable_r], mat[r]);
		else if(mat[r][c] == 0){
			if(IsInvalidEqu(c, mat[r]))
				return NO_SOLUTION;
			--r, status = INFINITE_SOLUTION;	// c-th variable can be anything
			// Actually NOT infinite. As independent variable can take p-1 values
			continue;
		}
		int modInv = pow(mat[r][c], p-2, p);

		for(int j = m; j >= c; --j) // convert diagonal to 1
			mat[r][j] = (mat[r][j] * modInv)%p;

		// zero all my column, except current row. Optimize for spare matrix
		for(int k = 0; k < n; ++k) if(k != r && mat[k][c] != 0) {
			for(int j = m; j >= c; --j)	// Add to kth-row: -mat[k][c] * mat[r] to zero it
			{
				mat[k][j] -= mat[k][c] * mat[r][j];
				mat[k][j] = (mat[k][j] % p + p) %p;
			}
		}
	}
	return status;
}

const int N = 5;

bool IsInvalidEqu(int c, bitset<N> &row) {
	// 0x+0y = 10 is invalid equ
	for(;c < (int)row.size()-1;++c) if(row[c] != 0)
		return false;

	return row[row.size()-1] == 0 ? false : true;
}

/*
vector< bitset<N> > binMat(4);
binMat[0] = bitset<N>(string("11111"));
binMat[1] = bitset<N>(string("01011"));
binMat[2] = bitset<N>(string("11101"));
binMat[3] = bitset<N>(string("00111"));

Note in bitset: First column here is the right hand side (1010)
				binMat[1][0] = 1
 */
GaussStatus GaussElimMod2(vector<bitset<N>> & mat){
	int n = mat.size(), m = mat[0].size()-1;
	GaussStatus status = UNIQUE;

	for(int r = 0, c = 0; r < n && c < m; ++r, ++c){
		int stable_r = r;
		for(int j = r; j < n; ++j)
			if(mat[j][c] != 0) {
				stable_r = j;
				break;
			}

		if(stable_r != r)
			swap(mat[stable_r], mat[r]);
		else if(mat[r][c] == 0){
			if(IsInvalidEqu(c, mat[r]))
				return NO_SOLUTION;
			--r, status = INFINITE_SOLUTION;	// c-th variable can be 0 or 1
			// Actually NOT infinite. If 3 columns can be 0/1, then # solutions are 2^3
			continue;
		}

		print(mat);

		// Subtraction under mod 2 is just xor
		for(int k = 0; k < n; ++k) if(k != r && mat[k][c] != 0)
			mat[k] ^= mat[r];
	}
	return status;
}

///////////// Determinant /////////////
ld determinant(vector<vector<ld> > mat) {
	ld ret = 1;
	for (int i = 0, j; i < sz(mat); i++) {
		for (j = i; j < sz(mat); j++)
			if ( !isZero(mat[j][i]))	break;
		if (j == sz(mat))	return 0;
		if (j != i)			ret = -ret, swap(mat[i], mat[j]);

		ret *= mat[i][i];
		ld inv = 1/mat[i][i];
		for (int k = i; k < sz(mat); k++)
			mat[i][k] *= inv;
		for (j = i + 1; j < sz(mat); j++) {
			for (int k = sz(mat) - 1; k >= i; k--)
				mat[j][k] -= mat[i][k] * mat[j][i];
		}
	}
	return ret;
}

//Find determinant%P
ll determinant(vector<vector<ll> > mat, ll P) {
	ll ret = 1;
	for (int i = 0, j; i < sz(mat); i++) {
		for (j = i; j < sz(mat); j++)
			if (mat[j][i] != 0) break;
		if (j == sz(mat)) return 0;
		if (j != i) 
			ret = -ret, ret = (ret%P+P)%P, swap(mat[i], mat[j]);

		(ret *= mat[i][i]) %= P;
		ll inv = modInverse(mat[i][i], P);
		for (int k = i; k < sz(mat); k++)
			(mat[i][k] *= inv) %= P;
		for (j = i + 1; j < sz(mat); j++)
			for (int k = sz(mat) - 1; k >= i; k--)
			{
				mat[j][k] -= mat[i][k] * mat[j][i];
				mat[j][k] = (mat[j][k]%P+P)%P;
			}
	}
	return (ret % P+P)%P;
}
```
# Catlan Number
- 1, 1, 2, 5, 14, 42, 132, 429, ....
- O(n^2).
- The Catalan number $C_n$  is the solution for: 
- Number of correct bracket sequence consisting of $n$  opening and $n$  closing brackets.
- The number of rooted full binary trees with  $n + 1$  leaves (vertices are not numbered). A rooted binary tree is full if every vertex has either two children or no children.
- The number of ways to completely parenthesize $n + 1$  factors.
- The number of triangulations of a convex polygon with $n + 2$  sides (i.e. the number of partitions of polygon into disjoint triangles by using the diagonals).
- The number of ways to connect the $2n$  points on a circle to form  $n$ disjoint chords.
- $$C_n = \binom{2n}{n} - \binom{2n}{n-1} = \frac{1}{n + 1} \binom{2n}{n} , {n} \geq 0$$ 
``` cpp
const int MOD = ....
const int MAX = ....
int catalan[MAX];
void init() {
    catalan[0] = catalan[1] = 1;
    for (int i=2; i<=n; i++) {
        catalan[i] = 0;
        for (int j=0; j < i; j++) {
            catalan[i] += (catalan[j] * catalan[i-j-1]) % MOD;
            if (catalan[i] >= MOD) {
                catalan[i] -= MOD;
            }
        }
    }
}
```
## Number of balanced sequences
- The number of balanced bracket sequences with only one bracket type can be calculated using the Catalan numbers. The number of balanced bracket sequences of length $2n$  ($n$  pairs of brackets) is: $$\frac{1}{n+1} \binom{2n}{n}$$ 
- If we allow $k$  types of brackets, then each pair be of any of the $k$  types (independently of the others), thus the number of balanced bracket sequences is:
$$\frac{1}{n+1} \binom{2n}{n} k^n$$ 

# Combinatorics 

- calculate until $1000*1000$ in $O(1)$
- pascal triangle generation.

``` c++
struct combination {
	vector<vector<ll>>C;
	int n;
	combination(int n = 1001, ll Mod = Mod) {
		this->n = n;
		C = vector<vector<ll>>(n, vector<ll>(n));
		init();
	}
	void init() {
		C[0][0] = 1;
		for (int i = 1; i < n; i++) {
			C[i][0] = C[i][i] = 1;
			for (int j = 1; j < i; j++)
				C[i][j] = (C[i - 1][j] % Mod + C[i - 1][j - 1] % Mod) % Mod;
		}
	}
	ll nCr(ll n, ll r) {
		return C[n][r];
	}
};
```

# Stars & Bars
- The number of ways to put $n$  identical objects into $k$  labeled boxes is: $$\binom{n + k - 1}{n}.$$
- Number of non-negative integer sums $$x_1 + x_2 + \dots + x_k = n$$  with  $x_i \ge 0$. The solution is $\binom{n + k - 1}{n}$.
- Number of positive integer sums $$x_1 + x_2 + \dots + x_k = n$$  with  $x_i \g 0$. The solution is $\binom{n - 1}{k - 1}$.

- calculate until $1e6$ in $O(log(n))$
- can calculate in $O(1)$ if Mod is smaller than 1e6
- $nCk = nC(n-k)$
- $nCk = (n/k)*[(n-1)C(k-1)]$
- ${sum k = [0..n] (nCk) }= 2^n$
- $nC1 + nC2 + .... + nCn$ = $2^n$
- `{sum k=[0=>m]((n+k)Ck)} = (n+m+1)Cm`
- `{sum m= [0=>n](mCk)} = (n+1)C(k+1)`
- $0Ck + 1Ck + .... + nCk = (n+1)C(k+1)$
- $nC0 + (n+1)C1 + (n+2)C2 + (n+3)C3= (n+4)C3$
- $(nC1) + 2 (nC2)+ ... + n(nCn) = n*2^n / 2 $
- $(nC0)^2 + (nC1)^2 + ... + (nCn)^2 = (2nCn)$
- $nC0 + (n-1)C1 + (n-2)C2 + ... 0Cn = Fibonacci(n+1)$
- The number of ways to place $k$ identical items into $n$ different places when any place can contain any amount of items is the definition of the number of k-combinations with repetitions: $$ans = C[n+k-1][k]$$
- In pascal triangle, the number of odd elements in row i = 2 ^ (#1's in the binary representation of i)
``` c++

struct comb {
	vector<ll>fc, invfc, fib, dr;
	int n;
	ll MOD;
	comb(int n = 1e6, ll Mod = Mod) {
		this->n = n;
		MOD = Mod;
		fc.assign(n, 0);
		fib.assign(n, 0);
		dr.assign(n, 0);
		invfc.assign(n, 0);
		factClc();
		//inverseClc(); //can only be used if MOD < 1e6
	}
	void factClc() {
		fc[0] = 1;
		for (int i = 1; i < n; i++)
			fc[i] = (fc[i - 1LL] % MOD * i % MOD) % MOD;
	}
	void inverseClc() {
		invfc[1] = 1;
		for (int i = 2; i < MOD; i++)
			invfc[i] = MOD - (MOD / i) * invfc[MOD % i] % MOD;
	}
	ll nCr(ll n, ll r) {
		return (fc[n] * inv(fc[r] * fc[n - r] % MOD) + MOD) % MOD;
	}
	ll nCrFast(ll n, ll r) {

		return (fc[n] * invfc[fc[r]] % MOD * invfc[fc[n - r]] % MOD + MOD) % MOD;
	}
	//only for small range ==> can overflow
	ll getC(ll n, int r)
	{
		if (r > n) return 0;
		ll res = 1;
		for (int i = 0; i < r; i++)
		{
			res *= n - i;
			res /= i + 1;
		}
		return res;
	}
	ll modpow(ll base, ll pow) {
		if (pow == 0)return 1 % MOD;
		ll u = modpow(base, pow / 2);
		u = (u * u) % MOD;
		if (pow % 2 == 1)u = (u * base) % MOD;
		return u;
	}
	ll add(ll a, ll b) {
		return ((a % MOD + b % MOD) % MOD + MOD) % MOD;
	}
	ll mul(ll a, ll b) {
		return ((a % MOD * b % MOD) % MOD + MOD) % MOD;
	}
	void derrangement() {
		dr[0] = 1;
		dr[1] = 0;
		for (int i = 2; i < n; i++)
			dr[i] = mul((i - 1LL), add(dr[i - 1], dr[i - 2]));
	}
	void fibonacci() {
		fib[0] = 1;
		fib[1] = 2;
		for (int i = 2; i < n; i++)
			fib[i] = add(fib[i - 1] , fib[i - 2]);
	}
};
```
### dearrangement

- Derrangement: permutation has no fixed point 
-	`[4 1 2 3]`: derragment
-	`[1 3 4 2]`: $NOT$  a derrangement (1 maps to 1)
-	$!n$ : number of derrangements for permutation of size n
-	note: $n!$ ==> factorial(n)  , $!n$ ==>derrangment(n)
-	$!n = [n/e]$ , $e = 2.71828...$ nearest integer to fraction
-	$!n = |_ n/e + 1/2 \_|$     ==> floor function
-	probabilty of no fixed points = $!n / n!  = n/e/n = 1/e = 0.37$
-	when $n>=4$ probability of derrangement  = 37% 
	or 63% to get a match (there are fixed points in permutaion)
-	`n:  0 1 2 3 4 5    6    7`
-	`!n: 1 0 1 2 9 44  265  1854`
-	dp recurrance: 
		$!n = (n-1) * [!(n-1)+!(n-2)] for\ n>=2$
		$!1 = 0, !0 = 1$
-	other recurrance:
	$!n =  n==0? 1 : n * !(n-1) + (-1)^ n$

# matrix exp
```cpp
#include <bits/stdc++.h>
 
using namespace std;
 
#define ll long long
#define IO ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
 int MOD=1e9+7;
#define REP(i,n) for(int i = 0; i < (n); i++)
struct Matrix {
	ll a[2][2] = {{0,0},{0,0}};
	Matrix operator *(const Matrix& other) {
		Matrix product;
		REP(i,2)REP(j,2)REP(k,2) {
			product.a[i][k] += a[i][j] * other.a[j][k]%MOD;
			product.a[i][k]%=MOD;
		}
		return product;
	}
};
Matrix expo_power(Matrix a, ll k) {
	Matrix product;
	REP(i,2) product.a[i][i] = 1;
	while(k > 0) {
		if(k % 2) {
			product = product * a;
		}
		a = a * a;
		k /= 2;
	}
	return product;
}
void solve()
{
    LDBL_MAX n, m;
    cin >> n;
    Matrix ree;
    ree.a[0][0]=19;
    ree.a[0][1]=7;
    ree.a[1][0]=6;
    ree.a[1][1]=20;
    Matrix z= expo_power(ree,n);
    // printMat(z);
    cout<<z.a[0][0]<<'\n';
    return;
}
```
# FFT
```cpp
using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd> & a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}


const int mod = 7340033;
const int root = 5;
const int root_1 = 4404020;
const int root_pw = 1 << 20;

void fft(vector<int> & a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        int wlen = invert ? root_1 : root;
        for (int i = len; i < root_pw; i <<= 1)
            wlen = (int)(1LL * wlen * wlen % mod);

        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; j++) {
                int u = a[i+j], v = (int)(1LL * a[i+j+len/2] * w % mod);
                a[i+j] = u + v < mod ? u + v : u + v - mod;
                a[i+j+len/2] = u - v >= 0 ? u - v : u - v + mod;
                w = (int)(1LL * w * wlen % mod);
            }
        }
    }

    if (invert) {
        int n_1 = inverse(n, mod);
        for (int & x : a)
            x = (int)(1LL * x * n_1 % mod);
    }
}

vector<int> multiply(vector<int> const& a, vector<int> const& b) {
    vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < a.size() + b.size()) 
        n <<= 1;
    fa.resize(n);
    fb.resize(n);

    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, true);

    vector<int> result(n);
    for (int i = 0; i < n; i++)
        result[i] = round(fa[i].real());
    return result;
}
 int carry = 0;
    for (int i = 0; i < n; i++)
        result[i] += carry;
        carry = result[i] / 10;
        result[i] %= 10;
    }

```
# pentagonal number
```cpp
//pentagonal numbers and ways to partition

ll penta[N];
ll getNthPentagonalNumber(ll n)
{
  n = penta[n];
  return (3ll * n * n - n) / 2;
}
ll dp[N]; // this dp is the number of ways to partition a number using pentagonal numbers
ll solve(ll n)
{
  if (n < 0)
    return 0;
  if (n == 0)
    return 1;
  ll &ret = dp[n];
  if (~ret)
    return ret;
  ret = 0;
  for (int i = 1, z = 1;; i += 2, z++)
  {
    if (z % 2)
    {
      ret = (ret + solve(n - getNthPentagonalNumber(i))) % MOD;
      ret = (ret + solve(n - getNthPentagonalNumber(i + 1))) % MOD;
    }
    else
    {
      ret = (ret - solve(n - getNthPentagonalNumber(i)) + MOD) % MOD;
      ret = (ret - solve(n - getNthPentagonalNumber(i + 1)) + MOD) % MOD;
    }
    // cout << getNthPentagonalNumber(i) << " ss " << i << " " << n << " " << z << " " << ret << '\n';
    if (n - getNthPentagonalNumber(i + 1) <= 0)
      break;
  }
  return ret;
}
void init(){
	penta[0] = 0;
  int x = 0;
  memset(dp, -1, sizeof dp);
  for (int i = 0; i < N; i++)
  {
    penta[i] = x;
    x = abs(x);
    if (i == 0 || i % 2 == 0)
      x++;
    else
      x *= -1;
  }
  solve(N - 2);
}
```
# fib to the power k
```cpp
// sum of fib to the power k (every term)
#include <bits/stdc++.h>
using namespace std;
template<class T, class S>
ostream& operator << (ostream &o, const pair<T, S> &p) {
    return o << '(' << p.first << ", " << p.second << ')';
}
template<template<class, class...> class T, class... A>
typename enable_if<!is_same<T<A...>, string>(), ostream&>::type
operator << (ostream &o, T<A...> V) {
	o << '[';
	for(auto a : V) o << a << ", ";
	return o << ']';
}
 
typedef long long int ll;
typedef long double ld;
typedef pair<ll, ll> pl;
typedef vector<ll> vl;
 
#define G(x) ll x; cin >> x;
#define GD(x) ld x; cin >> x;
#define GS(s) string s; cin >> s;
#define F(i, l, r) for(ll i = l; i < (r); ++i)
#define FD(i, r, l) for(ll i = r; i > (l); --i)
#define P(a, n) { cout << "{ "; F(_, 0, n) cout << a[_] << " "; cout << "}\n"; }
#define EX(x) { cout << x << '\n'; exit(0); }
#define A(a) (a).begin(), (a).end()
#define U first
#define R second
#define M 1000000007 //998244353
#define N 200010
#define NCR(n, r) (f[n] * fi[r] % M * fi[(n) - (r)] % M)
 
ll f[N], fi[N];
 
pl operator*(pl a, pl b) {
    return { (a.U * b.U + 5 * a.R * b.R) % M, (a.U * b.R + a.R * b.U) % M };
}
 
pl operator+(pl a, pl b) {
    return { (a.U + b.U) % M, (a.R + b.R) % M };
}
 
pl pw(pl a, ll p) { return p ? pw(a * a, p / 2) * (p & 1 ? a : pl{1, 0}) : pl{1, 0}; }
 
ll inv(ll a, ll b = M) { return 1 < a ? b - inv(b % a, a) * b / a : 1; } //inv a mod b
 
pl inv(pl p) { return pl{p.U, (M - p.R) % M} * pl{inv((p.U * p.U + M - 5 * p.R * p.R % M) % M), 0}; }
 
int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    f[0] = fi[0] = 1;
    F(i, 1, N) f[i] = i * f[i - 1] % M, fi[i] = inv(f[i]);
    G(n) G(k)
    pl ans = {0, 0};
    F(i, 0, k + 1) {
        pl p = pw({1, 1}, i) * pw({1, M - 1}, k - i) * pw({inv(2), 0}, k);
        pl q = (p == pl{1, 0} ? pl{(n + 1) % M, 0} : (pw(p, n + 1) + pl{M - 1, 0}) * inv(p + pl{M - 1, 0}));
        ans = ans + q * pl{NCR(k, i) * ((k - i) % 2 ? M - 1 : 1) % M, 0};
    }
    cout << (ans * inv(pw({0, 1}, k))).U << '\n';
}
```

# seive
```cpp

vector<int> primes, is_prime, spf, mobius, phi;

void sieve(int n)
{
  primes.clear();
  is_prime.assign(n + 1, 1);
  spf.assign(n + 1, 0);
  mobius.assign(n + 1, 0);
  phi.assign(n + 1, 0);
  is_prime[0] = is_prime[1] = 0;
  mobius[1] = phi[1] = 1;
  for (ll i = 2; i <= n; i++)
  {
    if (is_prime[i])
    {
      primes.push_back(i);
      spf[i] = i;
      mobius[i] = -1;
      phi[i] = i - 1;
    }
    for (auto p : primes)
    {
      if (i * p > n || p > spf[i])
        break;
      is_prime[i * p] = 0;
      spf[i * p] = p;
      mobius[i * p] = (spf[i] == p) ? 0 : -mobius[i];
      phi[i * p] = (spf[i] == p) ? phi[i] * p : phi[i] * phi[p];
    }
  }
}
ll phi(ll n)
{
  ll result = n;
  for (ll i = 2; i * i <= n; i++)
  {
    if (n % i == 0)
    {
      while (n % i == 0)
        n /= i;
      result -= result / i;
    }
  }
  if (n > 1)
    result -= result / n;
  return result;
}

```


### strirling
- n! ≈sqrt(2*pi*n)*(n^n)*(e^(-n)) stirling approximation.


# Factorial factorization

- get number of copies of a prime in a factiorial
- Given n and p what is the max x such n%(p^x)==0

```c++
int FactN_primePower(int n,int p){//O(log n base p)   ||| n should be the factorial value
	int pow=0;
	for(int i=p;i<=n;i*=p){
		pow+=n/i;
	}
}
```
## gray code 
- every two successive numbers have a difference of one bit 
```c++
ll grayCode(ll i){
	return (i ^ (i>>1));
}
void printGrayCode(ll len){
	for(int i=0;i< (1<<len)-1;i++){
		bitset<4>a(i);
		cout<< a <<"\t\t";
		a = grayCode(i);
		cout<< a<<"\t\t";
		cout<<__builtin_popcount(grayCode(i))<<endl; 
	}
}
```
# Placing Bishops on a Chessboard¶
- Find the number of ways to place $K$  bishops on an $N \times N$  chessboard so that no two bishops attack each other.
```cpp
int squares (int i) {
    if (i & 1)
        return i / 4 * 2 + 1;
    else
        return (i - 1) / 4 * 2 + 2;
}
int bishop_placements(int N, int K)
{
    if (K > 2 * N - 1)
        return 0;

    vector<vector<int>> D(N * 2, vector<int>(K + 1));
    for (int i = 0; i < N * 2; ++i)
        D[i][0] = 1;
    D[1][1] = 1;
    for (int i = 2; i < N * 2; ++i)
        for (int j = 1; j <= K; ++j)
            D[i][j] = D[i-2][j] + D[i-2][j-1] * (squares(i) - j + 1);

    int ans = 0;
    for (int i = 0; i <= K; ++i)
        ans += D[N*2-1][i] * D[N*2-2][K-i];
    return ans;
}
```
