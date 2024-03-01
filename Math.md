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

<h3> OR: just get min sum when shift_solution is called every time</h3>

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
- Complexity:
1- Search and reshuffle the pivoting row. This takes $O(n + m)$  when using heuristic mentioned above.
2- If the pivot element in the current column is found - then we must add this equation to all other equations, which takes time $O(nm)$ .
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