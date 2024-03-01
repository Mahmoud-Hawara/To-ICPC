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