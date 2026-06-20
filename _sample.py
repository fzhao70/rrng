"""R's sample.int algorithms (src/main/random.c, src/main/sort.c), bit-for-bit.

All functions take the RRNG `g` for stream draws and return 0-based int64 indices.
Equal-prob with replacement lives in RRNG.sample_index (vectorized); the rest is here:
  - sample_noreplace  : equal prob, Fisher-Yates + R_unif_index
  - sample_prob       : weighted, dispatching exactly as R's do_sample
      * ProbSampleReplace      (cumulative, replace or size<2, few heavy cells)
      * walker_ProbSampleReplace (alias method, when >200 cells have n*p > 0.1)
      * ProbSampleNoReplace    (sequential mass removal)
"""
import math

import numpy as np


def revsort(a, ib, n):
    """R's revsort: heapsort a[0:n] DESCENDING, carrying ib[] along. In-place.

    Faithful 1-indexed port of src/main/sort.c so equal-key order matches R exactly.
    """
    if n <= 1:
        return
    # emulate R's `a--; ib--;` 1-based indexing via helpers
    def A(i):
        return a[i - 1]

    def setA(i, v):
        a[i - 1] = v

    def B(i):
        return ib[i - 1]

    def setB(i, v):
        ib[i - 1] = v

    l = (n >> 1) + 1
    ir = n
    while True:
        if l > 1:
            l -= 1
            ra = A(l)
            ii = B(l)
        else:
            ra = A(ir)
            ii = B(ir)
            setA(ir, A(1))
            setB(ir, B(1))
            ir -= 1
            if ir == 1:
                setA(1, ra)
                setB(1, ii)
                return
        i = l
        j = l << 1
        while j <= ir:
            if j < ir and A(j) > A(j + 1):
                j += 1
            if ra > A(j):
                setA(i, A(j))
                setB(i, B(j))
                i = j
                j += i
            else:
                j = ir + 1
        setA(i, ra)
        setB(i, ii)


def _normalize_prob(prob, n):
    p = np.array(prob, dtype=np.float64)
    if p.size != n:
        raise ValueError(f"prob length {p.size} != n {n}")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("prob must be finite and non-negative")
    s = p.sum()
    if s == 0:
        raise ValueError("prob sums to zero")
    return p / s


def sample_noreplace(g, n, size):
    """Equal-prob sample(n, size, replace=FALSE): Fisher-Yates with R_unif_index."""
    if size > n:
        raise ValueError("cannot take a sample larger than the population when replace=FALSE")
    x = list(range(n))
    m = n
    out = np.empty(size, dtype=np.int64)
    for i in range(size):
        j = g._unif_index(m)
        out[i] = x[j]
        x[j] = x[m - 1]
        m -= 1
    return out


def _prob_sample_replace(g, n, p, nans):
    perm = list(range(n))                 # 0-based identities
    p = list(p)
    revsort(p, perm, n)
    for i in range(1, n):                 # cumulative
        p[i] += p[i - 1]
    nm1 = n - 1
    out = np.empty(nans, dtype=np.int64)
    for i in range(nans):
        rU = g.unif_rand()
        j = 0
        while j < nm1:
            if rU <= p[j]:
                break
            j += 1
        out[i] = perm[j]
    return out


def _prob_sample_noreplace(g, n, p, nans):
    perm = list(range(n))                 # 0-based identities
    p = list(p)
    revsort(p, perm, n)
    out = np.empty(nans, dtype=np.int64)
    totalmass = 1.0
    n1 = n - 1
    for i in range(nans):
        rT = totalmass * g.unif_rand()
        mass = 0.0
        j = 0
        while j < n1:
            mass += p[j]
            if rT <= mass:
                break
            j += 1
        out[i] = perm[j]
        totalmass -= p[j]
        for k in range(j, n1):
            p[k] = p[k + 1]
            perm[k] = perm[k + 1]
        n1 -= 1
    return out


def _walker_prob_sample_replace(g, n, p, nans):
    """R's Walker alias method (used when >200 cells have n*p > 0.1)."""
    q = [0.0] * n
    a = [0] * n
    HL = [0] * n
    H = -1            # H grows upward from start of HL (index of last pushed)
    L = n             # L shrinks downward from end of HL (index one past last pushed low end)
    for i in range(n):
        q[i] = p[i] * n
        if q[i] < 1.0:
            H += 1
            HL[H] = i
        else:
            L -= 1
            HL[L] = i
    if H >= 0 and L < n:                  # some q >= 1 and some < 1
        for k in range(n - 1):
            i = HL[k]
            j = HL[L]
            a[i] = j
            q[j] += q[i] - 1.0
            if q[j] < 1.0:
                L += 1
            if L >= n:
                break
    for i in range(n):
        q[i] += i
    out = np.empty(nans, dtype=np.int64)
    rounding = g.sample_kind == "rounding"
    for i in range(nans):
        if rounding:
            rU = g.unif_rand() * n
            k = int(rU)
        else:                                # R >= 3.6 default: index + separate coin
            k = g._unif_index(n)
            rU = k + g.unif_rand()
        out[i] = k if rU < q[k] else a[k]
    return out


def sample_prob(g, n, size, replace, prob):
    p = _normalize_prob(prob, n)
    if replace:                              # R do_sample: if (replace) {...} else {...}
        nc = int(np.sum(n * p > 0.1))
        if nc > 200:
            return _walker_prob_sample_replace(g, n, p, size)
        return _prob_sample_replace(g, n, p, size)
    if size > n:
        raise ValueError("cannot take a sample larger than the population when replace=FALSE")
    return _prob_sample_noreplace(g, n, p, size)
