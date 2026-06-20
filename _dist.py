"""R's nmath non-uniform variate generators, bit-for-bit.

Faithful Python ports of R's src/nmath: exp_rand/rexp (Ahrens-Dieter 1972), rpois
(Ahrens-Dieter 1982), rbinom (Kachitvichyanukul-Schmeiser BTPE 1988 + inversion),
rgamma (Ahrens-Dieter GD for shape>=1, GS for shape<1). Each takes the RRNG `g` and
draws from its stream via g.unif_rand() / g.norm_rand().

R caches per-parameter setup in C statics; that setup consumes no RNG and depends only
on the parameters, so we recompute it each call -- identical values, identical draws.
"""
import math

M_1_SQRT_2PI = 0.398942280401432677939946059934   # 1/sqrt(2*pi)


def fsign(x, y):
    return math.fabs(x) if y >= 0 else -math.fabs(x)


def R_pow_di(x, n):
    """R's R_pow_di: x**n for integer n via repeated squaring (bit-for-bit)."""
    pow_ = 1.0
    if n != 0:
        is_neg = n < 0
        if is_neg:
            n = -n
        while True:
            if n & 1:
                pow_ *= x
            n >>= 1
            if n:
                x *= x
            else:
                break
        if is_neg:
            pow_ = 1.0 / pow_
    return pow_


# ------------------------------- exp_rand / rexp ------------------------------
_Q = (
    0.6931471805599453, 0.9333736875190459, 0.9888777961838675, 0.9984959252914960,
    0.9998292811061389, 0.9999833164100727, 0.9999985691438767, 0.9999998906925558,
    0.9999999924734159, 0.9999999995283275, 0.9999999999728814, 0.9999999999985598,
    0.9999999999999289, 0.9999999999999968, 0.9999999999999999, 1.0000000000000000,
)


def exp_rand(g):
    """One standard exponential variate (R's exp_rand, Ahrens-Dieter)."""
    a = 0.0
    u = g.unif_rand()
    while u <= 0.0 or u >= 1.0:
        u = g.unif_rand()
    while True:
        u += u
        if u > 1.0:
            break
        a += _Q[0]
    u -= 1.0
    if u <= _Q[0]:
        return a + u
    i = 0
    ustar = g.unif_rand()
    umin = ustar
    while True:
        ustar = g.unif_rand()
        if umin > ustar:
            umin = ustar
        i += 1
        if u <= _Q[i]:
            break
    return a + umin * _Q[0]


def rexp(g, scale):
    """One Exp(scale) variate; R's rexp(scale=1/rate) = scale * exp_rand()."""
    return scale * exp_rand(g)


# ------------------------------------ rpois ----------------------------------
_PA0, _PA1, _PA2, _PA3 = -0.5, 0.3333333, -0.2500068, 0.2000118
_PA4, _PA5, _PA6, _PA7 = -0.1661269, 0.1421878, -0.1384794, 0.1250060
_ONE_7 = 0.1428571428571428571
_ONE_12 = 0.0833333333333333333
_ONE_24 = 0.0416666666666666667
_FACT = (1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.)


def rpois(g, mu):
    """One Poisson(mu) variate (R's rpois, Ahrens-Dieter 1982). Returns a float count."""
    if not math.isfinite(mu) or mu < 0:
        return float("nan")
    if mu <= 0.0:
        return 0.0

    big_mu = mu >= 10.0
    if not big_mu:
        # ---- small mu: cumulative-table inversion (one unif per outer pass) ----
        m = max(1, int(mu))
        l = 0
        q = p0 = p = math.exp(-mu)
        pp = [0.0] * 36
        while True:
            u = g.unif_rand()
            if u <= p0:
                return 0.0
            if l != 0:
                kstart = 1 if u <= 0.458 else min(l, m)
                for k in range(kstart, l + 1):
                    if u <= pp[k]:
                        return float(k)
                if l == 35:
                    continue
            l += 1
            done = False
            for k in range(l, 36):
                p *= mu / k
                q += p
                pp[k] = q
                if u <= q:
                    l = k
                    done = True
                    ret = float(k)
                    break
            if done:
                return ret
            l = 35

    # ---- big mu (>=10): PTRS-like rejection ----
    s = math.sqrt(mu)
    d = 6.0 * mu * mu
    big_l = math.floor(mu - 1.1484)
    omega = M_1_SQRT_2PI / s
    b1 = _ONE_24 / mu
    b2 = 0.3 * b1 * b1
    c3 = _ONE_7 * b1 * b2
    c2 = b2 - 15.0 * c3
    c1 = b1 - 6.0 * b2 + 45.0 * c3
    c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3
    c = 0.1069 / mu

    def step_F(pois, fk, difmuk, kflag, E, u):
        if pois < 10:
            px = -mu
            py = math.pow(mu, pois) / _FACT[int(pois)]
        else:
            del_ = _ONE_12 / fk
            del_ = del_ * (1.0 - 4.8 * del_ * del_)
            v = difmuk / fk
            if math.fabs(v) <= 0.25:
                px = fk * v * v * (((((((_PA7 * v + _PA6) * v + _PA5) * v + _PA4) *
                                      v + _PA3) * v + _PA2) * v + _PA1) * v + _PA0) - del_
            else:
                px = fk * math.log(1.0 + v) - difmuk - del_
            py = M_1_SQRT_2PI / math.sqrt(fk)
        x = (0.5 - difmuk) / s
        x *= x
        fx = -0.5 * x
        fy = omega * (((c3 * x + c2) * x + c1) * x + c0)
        if kflag > 0:
            return c * math.fabs(u) <= py * math.exp(px + E) - fy * math.exp(fx + E)
        return fy - u * fy <= py * math.exp(px - fx)

    g_val = mu + s * g.norm_rand()
    if g_val >= 0.0:
        pois = math.floor(g_val)
        if pois >= big_l:
            return float(pois)
        fk = float(pois)
        difmuk = mu - fk
        u = g.unif_rand()
        if d * u >= difmuk * difmuk * difmuk:
            return float(pois)
        if step_F(pois, fk, difmuk, 0, 0.0, u):     # kflag = 0, goto Step_F
            return float(pois)

    while True:
        E = exp_rand(g)
        u = 2.0 * g.unif_rand() - 1.0
        t = 1.8 + fsign(E, u)
        if t > -0.6744:
            pois = math.floor(mu + s * t)
            fk = float(pois)
            difmuk = mu - fk
            if step_F(pois, fk, difmuk, 1, E, u):    # kflag = 1
                return float(pois)


# ------------------------------------ rbinom ---------------------------------
def rbinom(g, nin, pp):
    """One Binomial(nin, pp) variate (R's rbinom, BTPE + inversion). Float count."""
    r = float(round(nin))                            # R_forceint
    if not math.isfinite(pp) or r < 0 or pp < 0.0 or pp > 1.0:
        return float("nan")
    if r == 0 or pp == 0.0:
        return 0.0
    if pp == 1.0:
        return r
    n = int(r)

    p = min(pp, 1.0 - pp)
    q = 1.0 - p
    np_ = n * p
    rr = p / q
    gg = rr * (n + 1)

    if np_ < 30.0:
        # ---- inversion for small mean ----
        qn = R_pow_di(q, n)
        while True:
            ix = 0
            f = qn
            u = g.unif_rand()
            broke = False
            while True:
                if u < f:
                    broke = True
                    break
                if ix > 110:
                    break
                u -= f
                ix += 1
                f *= (gg / ix - rr)
            if broke:
                break
    else:
        # ---- BTPE setup ----
        ffm = np_ + p
        m = int(ffm)
        fm = float(m)
        npq = np_ * q
        p1 = float(int(2.195 * math.sqrt(npq) - 4.6 * q)) + 0.5
        xm = fm + 0.5
        xl = xm - p1
        xr = xm + p1
        c = 0.134 + 20.5 / (15.3 + fm)
        al = (ffm - xl) / (ffm - xl * p)
        xll = al * (1.0 + 0.5 * al)
        al = (xr - ffm) / (xr * q)
        xlr = al * (1.0 + 0.5 * al)
        p2 = p1 * (1.0 + c + c)
        p3 = p2 + c / xll
        p4 = p3 + c / xlr

        while True:
            u = g.unif_rand() * p4
            v = g.unif_rand()
            if u <= p1:
                ix = int(xm - p1 * v + u)
                break
            if u <= p2:
                x = xl + (u - p1) / c
                v = v * c + 1.0 - math.fabs(xm - x) / p1
                if v > 1.0 or v <= 0.0:
                    continue
                ix = int(x)
            else:
                if u > p3:                           # right tail
                    ix = int(xr - math.log(v) / xlr)
                    if ix > n:
                        continue
                    v = v * (u - p3) * xlr
                else:                                # left tail
                    ix = int(xl + math.log(v) / xll)
                    if ix < 0:
                        continue
                    v = v * (u - p2) * xll
            k = abs(ix - m)
            if k <= 20 or k >= npq / 2 - 1:
                f = 1.0
                if m < ix:
                    for i in range(m + 1, ix + 1):
                        f *= (gg / i - rr)
                elif m != ix:
                    for i in range(ix + 1, m + 1):
                        f /= (gg / i - rr)
                if v <= f:
                    break
            else:
                amaxp = (k / npq) * ((k * (k / 3.0 + 0.625) + 0.1666666666666) / npq + 0.5)
                ynorm = -k * k / (2.0 * npq)
                alv = math.log(v)
                if alv < ynorm - amaxp:
                    break
                if alv <= ynorm + amaxp:
                    x1 = ix + 1
                    f1 = fm + 1.0
                    z = n + 1 - fm
                    w = n - ix + 1.0
                    z2 = z * z
                    x2 = x1 * x1
                    f2 = f1 * f1
                    w2 = w * w
                    if alv <= (xm * math.log(f1 / x1) + (n - m + 0.5) * math.log(z / w)
                               + (ix - m) * math.log(w * p / (x1 * q))
                               + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
                               + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
                               + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
                               + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0):
                        break

    if pp > 0.5:
        ix = n - ix
    return float(ix)


# ------------------------------------ rgamma ---------------------------------
_SQRT32 = 5.656854
_EXP_M1 = 0.36787944117144232159
_GQ1, _GQ2, _GQ3, _GQ4 = 0.04166669, 0.02083148, 0.00801191, 0.00144121
_GQ5, _GQ6, _GQ7 = -7.388e-5, 2.4511e-4, 2.424e-4
_GA1, _GA2, _GA3, _GA4 = 0.3333333, -0.250003, 0.2000062, -0.1662921
_GA5, _GA6, _GA7 = 0.1423657, -0.1367177, 0.1233795


def rgamma(g, a, scale):
    """One Gamma(shape=a, scale) variate (R's rgamma; GD for a>=1, GS for a<1)."""
    if a < 1.0:
        # ---- GS algorithm ----
        e = 1.0 + _EXP_M1 * a
        while True:
            p = e * g.unif_rand()
            if p >= 1.0:
                x = -math.log((e - p) / a)
                if exp_rand(g) >= (1.0 - a) * math.log(x):
                    break
            else:
                x = math.exp(math.log(p) / a)
                if exp_rand(g) >= x:
                    break
        return scale * x

    # ---- GD algorithm (a >= 1) ----
    s2 = a - 0.5
    s = math.sqrt(s2)
    d = _SQRT32 - s * 12.0

    t = g.norm_rand()
    x = s + 0.5 * t
    ret_val = x * x
    if t >= 0.0:
        return scale * ret_val

    u = g.unif_rand()
    if d * u <= t * t * t:
        return scale * ret_val

    r = 1.0 / a
    q0 = ((((((_GQ7 * r + _GQ6) * r + _GQ5) * r + _GQ4) * r + _GQ3) * r + _GQ2) * r + _GQ1) * r
    if a <= 3.686:
        b = 0.463 + s + 0.178 * s2
        si = 1.235
        c = 0.195 / s - 0.079 + 0.16 * s
    elif a <= 13.022:
        b = 1.654 + 0.0076 * s2
        si = 1.68 / s + 0.275
        c = 0.062 / s + 0.024
    else:
        b = 1.77
        si = 0.75
        c = 0.1515 / s

    def q_of(t):
        v = t / (s + s)
        if math.fabs(v) <= 0.25:
            return q0 + 0.5 * t * t * ((((((_GA7 * v + _GA6) * v + _GA5) * v + _GA4) * v
                                         + _GA3) * v + _GA2) * v + _GA1) * v
        return q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)

    if x > 0.0:
        if math.log(1.0 - u) <= q_of(t):
            return scale * ret_val

    while True:
        e = exp_rand(g)
        u = g.unif_rand()
        u = u + u - 1.0
        t = b - si * e if u < 0.0 else b + si * e
        if t >= -0.71874483771719:
            q = q_of(t)
            if q > 0.0:
                w = math.expm1(q)
                if c * math.fabs(u) <= w * math.exp(e - 0.5 * t * t):
                    break
    x = s + 0.5 * t
    return scale * x * x
