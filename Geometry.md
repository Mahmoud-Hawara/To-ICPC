# Point class

``` C++
template <class T>
struct Point
{
    T x, y;
    Point(T x, T y) : x(x), y(y){};
    Point &operator-=(Point p)
    {
        x -= p.x;
        y -= p.y;
        return *this;
    }
    Point &operator-(Point p)
    {
        return *(this) -= p;
    }
};
template <class T>
double length(Point<T> a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}
double fixAngle(double A)
{
    return A > 1 ? 1 : (A < -1 ? -1 : A);
}
```
## some basics

  - make sure to choose the right `intersect()` for planes or lines  

``` c++
template <class T>
struct Point
{
    T x, y, z;
    Point(T x, T y, T z = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    Point &operator+=(const Point &t)
    {
        x += t.x;
        y += t.y;
        z += t.z;
        return *this;
    }
    Point &operator-=(const Point &t)
    {
        x -= t.x;
        y -= t.y;
        z -= t.z;
        return *this;
    }
    Point &operator*=(T t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }
    Point operator/=(T t)
    {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }
    Point operator+(const Point &t) const
    {
        return Point(*this) += t;
    }
    Point operator-(const Point &t) const
    {
        return Point(*this) -= t;
    }
    Point operator*(T t) const
    {
        return Point(*this) *= t;
    }
    Point operator/(T t) const
    {
        return Point(*this) /= t;
    }
};
template <class T>
T dotProduct(Point<T> p1, Point<T> p2)
{
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}
template <class T>
T norm(const Point<T> p) // squared length a.a
{
    return dotProduct(p, p);
}
template <class T>
double Abs(Point<T> p) // length of the vector
{
    return sqrt(norm(p));
}
template <class T>
double projection(Point<T> p1, Point<T> p2) // projection of p1 into p2
{
    return dotProduct(p1, p2) / Abs(p2);
}
template <class T>
double angle(Point<T> p1, Point<T> p2) // angle between a, b
{
    return acos(dotProduct(p1, p2) / Abs(p1) / Abs(p2));
}
template <class T>
Point<T> cross(Point<T> p1, Point<T> p2)
{
    return Point<T>(p1.y * p2.z - p1.z * p2.y, p1.z * p2.x - p1.x * p2.z, p1.x * p2.y - p1.y * p2.x);
}
template <class T>
T triple(Point<T> p1, Point<T> p2, Point<T> p3)
{
    return dotProduct(p1, cross(p2, p3));
}
template <class T>
T intersect(Point<T> a1, Point<T> d1, Point<T> a2, Point<T> d2) // for lines
{
    return a1 + cross(a2 - a1, d2) / cross(d1, d2) * d1;
}
template <class T>
Point<T> intersect(Point<T> a1, Point<T> n1, Point<T> a2, Point<T> n2, Point<T> a3, Point<T> n3) // for planes
{
    Point<T> x(n1.x, n2.x, n3.x);
    Point<T> y(n1.y, n2.y, n3.y);
    Point<T> z(n1.z, n2.z, n3.z);
    Point<T> d(dotProduct(a1, n1), dotProduct(a2, n2), dotProduct(a3, n3));
    return Point<T>(triple(d, y, x), triple(x, d, z), triple(x, y, d)) / triple(n1, n2, n3);
}
```

# Triangle lows

  - side: S , Angle: A
  - low for side and two angles: Sin(A) / a = Sin(B) / b = Sin(C) / c
  - low for 3 sides and one angle: a*a = b*b + c*c - 2*b*c cos(A)

### get side from 1 side && 2 angles.
``` c++
double getSide_a_from_bAB(double b, double A, double B)
{
    return sin(A) * b / sin(B);
}
```

### get angle from 2 side && 1 angle
``` c++
double getAngle_A_from_abB(double a, double b, double B)
{
    return asin(fixAngle(a * sin(B) / b));
}

```

### get angle from 3 sides
``` c++
double getAngle_A_form_abc(double a, double b, double c)
{
    return acos(fixAngle((b * b + c * c - a * a) / (2 * a * b)));
}

```
## Triangle area
 - Given the length of three medains of triangle , find Area
 - A median of a triagle is a line segment join a vertex to the midpoint of the opposite side.

``` c++
double triangleAreaUsingMedains(double a, double b, double c)
{
    if (a <= 0 or b <= 0 or c <= 0)
        return -1;
    double s = 0.5 * (a + b + c);
    double medians_area = s * (s - a) * (s - b) * (s - c);
    double area = 4.0 / 3.0 * sqrt(medians_area);
    if (medians_area <= 0.0 or area <= 0.0)
        return -1;
    return area;
}
template <class T>
double triangleAreaUsingPoints(Point<T> a, Point<T> b, Point<T> c)
{
    double aa = length(a - b);
    double bb = length(b - c);
    double cc = length(c - a);
    double s = (aa + bb + cc) / 2;
    return sqrt(s * (s - aa) * (s - bb) * (s - cc));
}
```

# lines

- `det()`: get the determinate
- `equivalent(Line<T> m, Line<T> n)`: check if two lines are equivilant.
- `parallel(Line<T> n, Line<T> m)`: check if two lines are parallel.
- `bool intersect(Line<T> n, Line<T> m, Point<T> &res)`: check if two lines are intersect. the intersection point is in `res`

``` c++
#define EPS 0.000'000'000'001
template <class T>
struct Line
{
    T a, b, c;
    Line(){};
    Line(T a, T b, T c)
    {
        this->a = a;
        this->b = b;
        this->c = c;
    }
};
template <class T>
struct Point
{
    T x, y;
};
template <class T>
T det(T a, T b, T c, T d)
{
    return a * d - b * c;
}
template <class T>
bool equivalent(Line<T> m, Line<T> n)
{
    return abs(det(m.a, m.b, n.a, n.b)) < EPS && abs(det(m.a, m.c, n.a, n.c)) < EPS && abs(det(m.b, m.c, n.b, n.c)) < EPS;
}
template <class T>
bool parallel(Line<T> n, Line<T> m)
{
    return abs(det(n.a, n.b, m.a, m.b)) < EPS;
}
template <class T>
bool intersect(Line<T> n, Line<T> m, Point<T> &res)
{
    T zn = det(n.a, n.b, m.a, m.b);
    if (abs(zn) < EPS)
        return false;
    res.x = -det(n.c, n.b, m.c, m.b);
    res.y = -det(n.a, n.c, m.a, m.c);
    return true;
}
```


# Complex class in Geometry

``` c++
#include <bits/stdc++.h>
using namespace std;
#define IO ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0)
typedef long long ll;
const long long N = 3e5 + 7, Mod = 1e9 + 7, INF = 2e18;
ll inv(ll a, ll b = Mod) { return 1 < a ? b - inv(b % a, a) * b / a : 1; }
const int dx[9] = {0, 0, 1, -1, 1, 1, -1, -1, 0};
const int dy[9] = {1, -1, 0, 0, -1, 1, 1, -1, 0};
const double PI = acos(-1.0);
const double EPS = 1e-10;
typedef complex<double> point;
#define EPS 0.000'000'001
#define x real()
#define y imag()
#define angle(a) atan2(a.imag(), a.real())
#define vec(a, b) (b - a)
#define squaredLength(a) a.x *a.x + a.y *a.y
#define length(a) hypot(a.imag(), a.real()) // hypot is slower than sqrt but more accurate
#define normalize(a) a / length(a)
#define dotProduct(a, b) (conj(a) * b).real()
#define crossProduct(a, b) (conj(a) * b).imag()
#define rotate0(p, ang) p *exp(point(0, ang))
#define rotateAround(p, ang, about) rotate0(vec(about - p), ang) + about
#define reflect0(v, m) conj(v / m) * m
#define reflect(v, m, about) reflect0(vec(v, about), vec(m, about)) + about
#define norm(a) dotProduct(a, a)

bool isCollinear(point a, point b, point c)
{
    return fabs(crossProduct(vec(a, b), vec(a, c))) < EPS;
}
// Point C distance to line A-B
double distanceToLine(point a, point b, point c)
{
    double dist = crossProduct(vec(a, b), vec(a, c)) / length(vec(b, a));
    return fabs(dist);
}
// intersection of segment ab with segment cd
bool intersectSegments(point a, point b, point c, point d, point intersect)
{
    double d1 = crossProduct(a - b, d - c);
    double d2 = crossProduct(a - c, d - c);
    double d3 = crossProduct(a - b, a - c);

    if (fabs(d1) < EPS)
        return false; // parallel or idintical

    double t1 = d2 / d1;
    double t2 = d3 / d1;
    intersect = a + (b - a) * t1;

    if (t1 < -EPS or d2 < -EPS or t2 > 1 + EPS)
        return false; // ab is ray , cd is segment

    return true;
}
// where is p2 relative to segment p0-p1
// ccw = +1 => angle >0 or collinear after p1
// cw  = -1 => angle <0 or collinear after p0
// undefined = 0 => collinar in range [a,b]. Be carful here
int ccw(point p0, point p1, point p2)
{
    point v1(p1 - p0), v2(p2 - p0);
    if (crossProduct(v1, v2) > EPS)
        return +1;
    if (crossProduct(v1, v2) < -EPS)
        return -1;

    if (v1.x * v2.x < -EPS or v1.y * v2.y < -EPS)
        return -1;

    if (norm(v1) < norm(v2) - EPS)
        return +1;
    return 0;
}
bool intersect(point p1, point p2, point p3, point p4)
{
    // special case handling if a segment is just a point
    bool xx = (p1 == p2), yy = (p4 == p3);
    if (xx && yy)
        return p1 == p3;
    if (xx)
        return ccw(p3, p4, p2) == 0;

    if (yy)
        return ccw(p1, p2, p3) == 0;

    return ccw(p1, p2, p3) * ccw(p1, p2, p4) <= 0 and
           ccw(p3, p4, p1) * ccw(p3, p4, p2) <= 0;
}
// if a , b ,c are collinear => don't call this function
pair<double, point> findCircle(point a, point b, point c)
{
    point m1 = (b + a) * 0.5, v1 = b - a, pv1 = point(v1.y, -v1.x);
    point m2 = (b + c) * 0.5, v2 = b - c, pv2 = point(v2.y, -v2.x);
    point end1 = m1 + pv1, end2 = m2 + pv2, center;
    intersectSegments(m1, end1, m2, end2, center);
    double len = length(a);
    return make_pair(len, center);
}
int dcmp(double a, double b) {
	return (fabs(a - b) <= EPS ? 0 : (a < b ? -1 : 1));
}
// return 0,1 or 2 points => using parameteric parameters / substitution method
vector<point> intersectLineCircle(point p0, point p1, point C, double r)
{
    double a = dotProduct(p1 - p0, p1 - p0);
    double b = 2 * dotProduct(p1 - p0, p0 - C);
    double c = dotProduct(p0 - C, p0 - C) - r * r;
    double f = b * b - 4 * a * c;

    vector<point> v;
    if (dcmp(f, 0) >= 0)
    {
        if (dcmp(f, 0) == 0)
            f = 0;
        double t1 = (-b + sqrt(f)) / 2 * a;
        double t2 = (-b - sqrt(f)) / 2 * a;
        v.push_back(p0 + t1 * (p1 - p0));
        if (dcmp(f, 0) != 0)
            v.push_back(p0 + t2 * (p1 - p0));
    }
    return v;
}

```

# Point related to polygon

- O(n) implementation for each point.
``` c++

// Checking if a point is inside a polygon O(N) if a point is on one side its considered outside
bool point_in_polygon(Point point, vector<Point> polygon)
{
    int num_vertices = polygon.size();
    double x = point.x, y = point.y;
    bool inside = false;

    // Store the first point in the polygon and initialize
    // the second point
    Point p1 = polygon[0], p2;

    // Loop through each edge in the polygon
    for (int i = 1; i <= num_vertices; i++)
    {
        // Get the next point in the polygon
        p2 = polygon[i % num_vertices];

        // Check if the point is above the minimum y
        // coordinate of the edge
        if (y > min(p1.y, p2.y))
        {
            // Check if the point is below the maximum y
            // coordinate of the edge
            if (y <= max(p1.y, p2.y))
            {
                // Check if the point is to the left of the
                // maximum x coordinate of the edge
                if (x <= max(p1.x, p2.x))
                {
                    // Calculate the x-intersection of the
                    // line connecting the point to the edge
                    double x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;

                    // Check if the point is on the same
                    // line as the edge or to the left of
                    // the x-intersection
                    if (p1.x == p2.x || x <= x_intersection)
                    {
                        // Flip the inside flag
                        inside = !inside;
                    }
                }
            }
        }

        // Store the current point as the first point for
        // the next iteration
        p1 = p2;
    }

    // Return the value of the inside flag
    return inside;
}
```

- finding if a point leis inside a polygon (OR on any side) in log(N) .
- the polygon points must be in counter-clockwise order.

1- use `prepare()` </br>
2- use `pointInConvexPolygon()` for queries

``` c++
struct pt
{
    long long x, y;
    pt() {}
    pt(long long _x, long long _y) : x(_x), y(_y) {}
    pt operator+(const pt &p) const { return pt(x + p.x, y + p.y); }
    pt operator-(const pt &p) const { return pt(x - p.x, y - p.y); }
    long long cross(const pt &p) const { return x * p.y - y * p.x; }
    long long dot(const pt &p) const { return x * p.x + y * p.y; }
    long long cross(const pt &a, const pt &b) const { return (a - *this).cross(b - *this); }
    long long dot(const pt &a, const pt &b) const { return (a - *this).dot(b - *this); }
    long long sqrLen() const { return this->dot(*this); }
};

bool lexComp(const pt &l, const pt &r)
{
    return l.x < r.x || (l.x == r.x && l.y < r.y);
}

int sgn(long long val) { return val > 0 ? 1 : (val == 0 ? 0 : -1); }

vector<pt> seq;
pt translation;
int n, m;

bool pointInTriangle(pt a, pt b, pt c, pt point)
{
    long long s1 = abs(a.cross(b, c));
    long long s2 = abs(point.cross(a, b)) + abs(point.cross(b, c)) + abs(point.cross(c, a));
    return s1 == s2;
}
bool onSegment(pt s, pt e, pt p)
{
    return p.cross(s, e) == 0 and (s - p).dot(e - p) <= 0;
}
void prepare(vector<pt> &points)
{
    n = points.size();
    int pos = 0;
    for (int i = 1; i < n; i++)
    {
        if (lexComp(points[i], points[pos]))
            pos = i;
    }
    rotate(points.begin(), points.begin() + pos, points.end());

    n--;
    seq.resize(n);
    for (int i = 0; i < n; i++)
        seq[i] = points[i + 1] - points[0];
    translation = points[0];
}

bool pointInConvexPolygon(pt point)
{
    point = point - translation;
    if (seq[0].cross(point) != 0 &&
        sgn(seq[0].cross(point)) != sgn(seq[0].cross(seq[n - 1])))
        return false;
    if (seq[n - 1].cross(point) != 0 &&
        sgn(seq[n - 1].cross(point)) != sgn(seq[n - 1].cross(seq[0])))
        return false;
    /*
        considering the side is inside the polygon: 
        if (seq[0].cross(point) == 0)
        return seq[0].sqrLen() >= point.sqrLen();

        considering the side is inside the polygon: 
        if (seq[0].cross(point) == 0 || seq[n - 1].cross(point) == 0)
        return false;

    */
    if (seq[0].cross(point) == 0 || seq[n - 1].cross(point) == 0)
        return false;

    int l = 0, r = n - 1;
    while (r - l > 1)
    {
        int mid = (l + r) / 2;
        int pos = mid;
        if (seq[pos].cross(point) >= 0)
            l = mid;
        else
            r = mid;
    }
    int pos = l;


    // remove it if you want to consider the side to be inside the polygon
    if (onSegment(seq[pos], seq[pos + 1], point))
        return false;

    return pointInTriangle(seq[pos], seq[pos + 1], pt(0, 0), point);
}
void solve()
{
    cin >> n;
    vector<pt> v(n);
    for (int i = 0; i < n; i++)
        cin >> v[i].x >> v[i].y;

    reverse(v.begin(), v.end());

    prepare(v);

    cin >> m;
    bool ok = true;
    for (int i = 0; i < m; i++)
    {
        int x, y;
        cin >> x >> y;
        // cout << pointInConvexPolygon(pt(x, y)) << endl;
        if (!pointInConvexPolygon(pt(x, y)))
            ok = false;
    }
    cout << (ok ? "YES" : "NO") << endl;
}
```

# Convex Hull

### Graham's scan Algorithm
- get convex hull with collinear points : 3 pints in same line
- time : O(n log n)
- the convex hull points is returned in counter-clockwise order
- if you want to include collinear points => put the `include_collinear` to be true
otherwise it will be skipped
- you want to get smallest convex hull ? => don't include collinear points option.
- you should take care if a point is repeated 

``` c++

struct pt
{
    double x, y;
};

int orientation(pt a, pt b, pt c)
{
    double v = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
    if (v < 0)
        return -1; // clockwise
    if (v > 0)
        return +1; // counter-clockwise
    return 0;
}

bool cw(pt a, pt b, pt c, bool include_collinear)
{
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}
bool collinear(pt a, pt b, pt c) { return orientation(a, b, c) == 0; }

vector<pt> convex_hull(vector<pt> &a, bool include_collinear = false)
{
    pt p0 = *min_element(a.begin(), a.end(), [](pt a, pt b)
                         { return make_pair(a.y, a.x) < make_pair(b.y, b.x); });
    sort(a.begin(), a.end(), [&p0](const pt &a, const pt &b)
         {
        int o = orientation(p0, a, b);
        if (o == 0)
            return (p0.x-a.x)*(p0.x-a.x) + (p0.y-a.y)*(p0.y-a.y)
                < (p0.x-b.x)*(p0.x-b.x) + (p0.y-b.y)*(p0.y-b.y);
        return o < 0; });
    if (include_collinear)
    {
        int i = (int)a.size() - 1;
        while (i >= 0 && collinear(p0, a[i], a.back()))
            i--;
        reverse(a.begin() + i + 1, a.end());
    }

    vector<pt> st;
    for (int i = 0; i < (int)a.size(); i++)
    {
        while (st.size() > 1 && !cw(st[st.size() - 2], st.back(), a[i], include_collinear))
            st.pop_back();
        st.push_back(a[i]);
    }

    return st;
}
```

### Monotone chain Algorithm

``` c++
struct pt
{
    double x, y;
};

int orientation(pt a, pt b, pt c)
{
    double v = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
    if (v < 0)
        return -1; // clockwise
    if (v > 0)
        return +1; // counter-clockwise
    return 0;
}

bool cw(pt a, pt b, pt c, bool include_collinear)
{
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}
bool ccw(pt a, pt b, pt c, bool include_collinear)
{
    int o = orientation(a, b, c);
    return o > 0 || (include_collinear && o == 0);
}

vector<pt> convex_hull(vector<pt> a, bool include_collinear = false)
{
    if (a.size() == 1)
        return a;

    sort(a.begin(), a.end(), [](pt a, pt b)
         { return make_pair(a.x, a.y) < make_pair(b.x, b.y); });
    pt p1 = a[0], p2 = a.back();
    vector<pt> up, down;
    up.push_back(p1);
    down.push_back(p1);
    for (int i = 1; i < (int)a.size(); i++)
    {
        if (i == a.size() - 1 || cw(p1, a[i], p2, include_collinear))
        {
            while (up.size() >= 2 && !cw(up[up.size() - 2], up[up.size() - 1], a[i], include_collinear))
                up.pop_back();
            up.push_back(a[i]);
        }
        if (i == a.size() - 1 || ccw(p1, a[i], p2, include_collinear))
        {
            while (down.size() >= 2 && !ccw(down[down.size() - 2], down[down.size() - 1], a[i], include_collinear))
                down.pop_back();
            down.push_back(a[i]);
        }
    }

    if (include_collinear && up.size() == a.size())
    {
        reverse(a.begin(), a.end());
        return a;
    }
    a.clear();
    for (int i = 0; i < (int)up.size(); i++)
        a.push_back(up[i]);
    for (int i = down.size() - 2; i > 0; i--)
        a.push_back(down[i]);

    return a;
}
```


# Convex Hull Trick

- used for dp optimization
- can you transform the dp equation to mx + b ?
- use CHT for O(n log n).

```  c++
struct Line
{
    ll m, b;
    mutable function<const Line *()> succ;
    bool operator<(const Line &other) const
    {
        return m < other.m;
    }
    bool operator<(const ll &x) const
    {
        const Line *s = succ();
        if (!s)
            return 0;
        return b - s->b < (s->m - m) * x;
    }
};
// will maintain upper hull for maximum
struct CHT : public multiset<Line, less<>>
{
    bool bad(iterator y)
    {
        auto z = next(y);
        if (y == begin())
        {
            if (z == end())
                return 0;
            return y->m == z->m && y->b <= z->b;
        }
        auto x = prev(y);
        if (z == end())
            return y->m == x->m && y->b <= x->b;
        return (long double)(x->b - y->b) * (z->m - y->m) >= (long double)(y->b - z->b) * (y->m - x->m);
    }
    void insert_line(ll m, ll b)
    {
        auto y = insert({m, b});
        y->succ = [=]
        { return next(y) == end() ? 0 : &*next(y); };
        if (bad(y))
        {
            erase(y);
            return;
        }
        while (next(y) != end() && bad(next(y)))
            erase(next(y));
        while (y != begin() && bad(prev(y)))
            erase(prev(y));
    }

    ll query(ll x)
    {

        auto l = *lower_bound(x);
        return l.m * x + l.b;
    }
};
CHT cht;
ll n, c, h[N], dp[N];
// negative for minimizing...
void solve()
{
    cin >> n >> c;
    for (int i = 0; i < n; i++)
        cin >> h[i];

    cht.insert_line(2 * h[0], -(h[0] * h[0] + dp[0]));

    for (int i = 1; i < n; i++)
    {
        dp[i] = -cht.query(h[i]) + h[i] * h[i] + c;
        cht.insert_line(2 * h[i], -(h[i] * h[i] + dp[i]));
        // cout << dp[i] << " ";
    }
    cout << dp[n - 1] << endl;
}
```
  
# Area of simple polygon

This is easy to do if we go through all edges and add trapezoid areas bounded by each edge and x-axis. The area needs to be taken with sign so that the extra area will be reduced. Hence, the formula is as follows:

```c++
double area(const vector<point>& fig) {
    double res = 0;
    for (unsigned i = 0; i < fig.size(); i++) {
        point p = i ? fig[i - 1] : fig.back();
        point q = fig[i];
        res += (p.x - q.x) * (p.y + q.y);
    }
    return fabs(res) / 2;
}
```
# Regular polygon
- The sum of the interior angles = $(2n – 4)$ * $right Angles$
- interior angle of a regular polygon is $[(2n – 4) × 90°] / n$

### Area of regular polygon.
![Screenshot 2024-03-01 202155](https://github.com/Mahmoud-Hawara/To-ICPC/assets/66100565/ddf2394d-df00-44fd-87c5-7489e8ae8f5e)


![Screenshot 2024-03-01 202206](https://github.com/Mahmoud-Hawara/To-ICPC/assets/66100565/10b225bd-f9fb-42af-9013-2ef342242bc4)

