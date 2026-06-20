#!/usr/bin/env python3
"""Pure-Python replication of R's native Mersenne-Twister RNG.

Reproduces R's default generator bit-for-bit:  set.seed() + unif_rand() + runif()
+ sample(..., replace=TRUE)  (R >= 3.6 default 'Rejection' method, and the pre-3.6
'Rounding' method behind a flag).  NO R at runtime.  Validated against real R output.

FAST path: the MT core (the expensive part) is driven by NumPy's MT19937 SEEDED TO
R's STATE.  NumPy's MT19937 == R's MT algorithm, so with the identical 624-word state
+ position it emits the identical uint32 stream.  R's unif_rand scaling and R's sample()
rejection method are applied on top.  Results are identical to the slow pure-Python
reference (kept in `_genrand` for auditability).

R chain (src/main/RNG.c):
  RNG_Init  : scramble seed 50x by (69069*s + 1), then fill the 625 i_seed words the
              same way; FixupSeeds sets mti = 624.
  unif_rand : fixup(MT_genrand() * i2_32m1),  i2_32m1 = 1 / (2^32 - 1)
  sample(n) : R_unif_index(n) via rbits(ceil(log2 n)) with rejection of draws >= n
              (R >= 3.6); or floor(n * unif_rand()) (R < 3.6, 'Rounding').
"""
import math

import numpy as np

from ._qnorm import qnorm
from . import _sample
from . import _dist

BIG = 134217728  # 2^27, R's norm_rand Inversion constant

N, M = 624, 397
MATRIX_A, UPPER_MASK, LOWER_MASK = 0x9908b0df, 0x80000000, 0x7fffffff
MASK32 = 0xffffffff
# R's MT_genrand scales the raw uint32 by 1/2^32 (src/nmath ... RNG.c), NOT 1/(2^32-1).
MT_SCALE = 2.3283064365386963e-10        # 1 / 2^32  (R's MT_genrand multiplier)
I2_32M1 = 2.328306437080797e-10          # 1 / (2^32 - 1), used only by R's fixup edges
_LO = 0.5 * I2_32M1                       # fixup: x <= 0      -> 0.5 / (2^32 - 1)
_HI = 1.0 - 0.5 * I2_32M1                 # fixup: 1 - x <= 0  -> 1 - 0.5 / (2^32 - 1)

_SAMPLE_KINDS = ("rejection", "rounding")


class RRNG:
    """A stateful R-compatible Mersenne-Twister stream.

    Parameters
    ----------
    seed : int
        The argument to R's ``set.seed(seed)``.
    sample_kind : {"rejection", "rounding"}
        Index method for :meth:`sample_index`.  ``"rejection"`` matches R >= 3.6
        (the current default); ``"rounding"`` matches R < 3.6.

    The object is stateful so a single seed stream can be threaded across calls /
    regions, exactly as an R script that does ``set.seed(s)`` once and then loops:
    the order of consumption matters and must match.
    """

    def __init__(self, seed, sample_kind="rejection"):
        if sample_kind not in _SAMPLE_KINDS:
            raise ValueError(f"sample_kind must be one of {_SAMPLE_KINDS}, got {sample_kind!r}")
        self.sample_kind = sample_kind
        self.set_seed(seed)

    # ------------------------------------------------------------------ seeding
    def set_seed(self, seed):
        """Re-seed the stream exactly as R's ``set.seed(seed)``."""
        seed = int(seed) & MASK32
        for _ in range(50):                      # RNG_Init initial scrambling
            seed = (69069 * seed + 1) & MASK32
        iseed = [0] * 625
        for j in range(625):                     # fill mti + 624 state words
            seed = (69069 * seed + 1) & MASK32
            iseed[j] = seed
        iseed[0] = 624                           # FixupSeeds: mti = N
        self.mti = iseed[0]
        self.mt = iseed[1:625]
        # --- fast NumPy MT19937 seeded to R's exact state ---
        self._bg = np.random.MT19937()
        st = self._bg.state
        st['state']['key'][:] = np.asarray(self.mt, dtype=np.uint32)
        st['state']['pos'] = self.mti
        self._bg.state = st
        self._buf = None
        self._pos = 0
        return self

    # ---------------- pure-Python reference MT (validation only, slow) ----------
    def _genrand(self):
        """Reference MT_genrand(); not used in the fast path, kept for auditing."""
        if self.mti >= N:
            mt, mag01 = self.mt, (0, MATRIX_A)
            for kk in range(N - M):
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(N - M, N - 1):
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK)
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self.mti = 0
        y = self.mt[self.mti]; self.mti += 1
        y ^= (y >> 11); y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000; y ^= (y >> 18)
        return y & MASK32

    # ---------------- fast uniform buffer via NumPy MT19937 (R state) -----------
    def _refill(self, n):
        raw = self._bg.random_raw(int(n)).astype(np.float64)
        u = raw * MT_SCALE                       # R's MT_genrand: raw / 2^32 -> [0, 1)
        np.clip(u, _LO, _HI, out=u)              # R fixup of the 0 / 1 edges
        self._buf = u
        self._pos = 0

    def unif_rand(self):
        """One R ``runif(1)`` draw as a Python float."""
        if self._buf is None or self._pos >= self._buf.size:
            self._refill(8192)
        v = self._buf[self._pos]; self._pos += 1
        return float(v)

    def runif(self, n):
        """A length-``n`` R ``runif(n)`` block as a float64 numpy array."""
        n = int(n)
        out = np.empty(n, dtype=np.float64)
        filled = 0
        while filled < n:
            if self._buf is None or self._pos >= self._buf.size:
                self._refill(max(8192, n - filled))
            avail = self._buf[self._pos:]
            take = min(avail.size, n - filled)
            out[filled:filled + take] = avail[:take]
            self._pos += take
            filled += take
        return out

    # ---------------------- normal: R's default Inversion rnorm -----------------
    def norm_rand(self):
        """One R ``rnorm(1)`` draw (standard normal), normal.kind = "Inversion".

        Consumes exactly TWO unif_rand() draws (R combines them for 53-bit precision).
        """
        u1 = self.unif_rand()
        u1 = float(int(BIG * u1)) + self.unif_rand()   # (int) truncates toward 0; u1>0
        return float(qnorm(u1 / BIG))

    def rnorm(self, n, mean=0.0, sd=1.0):
        """A length-``n`` R ``rnorm(n, mean, sd)`` block (Inversion), as float64 array.

        Matches R's consumption order: 2 unif draws per value, interleaved
        (u1_0, u2_0, u1_1, u2_1, ...).
        """
        n = int(n)
        block = self.runif(2 * n)
        u1 = np.floor(BIG * block[0::2]) + block[1::2]   # (int) cast == floor for u>0
        p = u1 / BIG
        return mean + sd * qnorm(p)

    # -------- R sample(seq_len(n), size, replace=TRUE) -> 0-based indices --------
    def sample_index(self, n, size):
        """0-based indices of ``sample(seq_len(n), size, replace=TRUE)`` in R.

        Honours ``self.sample_kind`` ("rejection" = R >= 3.6, "rounding" = R < 3.6).
        """
        size = int(size)
        if n <= 0:
            return np.zeros(size, dtype=np.int64)
        if self.sample_kind == "rounding":
            return self._sample_index_rounding(n, size)
        return self._sample_index_rejection(n, size)

    def _sample_index_rounding(self, n, size):
        u = self.runif(size)
        idx = np.floor(n * u).astype(np.int64)
        np.clip(idx, 0, n - 1, out=idx)          # guard the u==1 edge
        return idx

    def _sample_index_rejection(self, n, size):
        bits = (n - 1).bit_length()              # ceil(log2 n)
        if bits > 15:
            # R's rbits(bits) loops `for (n = 0; n <= bits; n += 16)`, i.e. it draws
            # (bits // 16 + 1) 16-bit uniforms per index -- so 2 draws already at
            # bits == 16. The vectorized fast path below assumes a SINGLE 16-bit draw,
            # valid only for bits <= 15 (n <= 32768). Fall back to a correct scalar
            # loop otherwise (rare for bootstrap resampling).
            return self._sample_index_rejection_scalar(n, size, bits)
        mask = (1 << bits) - 1
        out = np.empty(size, dtype=np.int64)
        filled = 0
        while filled < size:
            need = size - filled
            if self._buf is None or self._pos >= self._buf.size:
                self._refill(max(8192, need * 2))
            avail = self._buf[self._pos:]
            rb = (np.floor(avail * 65536.0).astype(np.int64)) & mask   # rbits(bits)
            acc = rb < n
            cum = np.cumsum(acc)
            if cum.size and cum[-1] >= need:
                idx = int(np.searchsorted(cum, need))                  # need-th accept
                consumed = idx + 1
                out[filled:filled + need] = rb[:consumed][acc[:consumed]][:need]
                self._pos += consumed
                filled += need
            else:
                taken = rb[acc]
                out[filled:filled + taken.size] = taken
                filled += int(taken.size)
                self._pos = self._buf.size       # consumed the whole buffer
        return out

    def _sample_index_rejection_scalar(self, n, size, bits):
        chunks = bits // 16 + 1                   # 16-bit draws per rbits() call
        out = np.empty(size, dtype=np.int64)
        for i in range(size):
            while True:
                v = 0
                for _ in range(chunks):
                    v = 65536 * v + int(np.floor(self.unif_rand() * 65536.0))
                v &= (1 << bits) - 1
                if v < n:
                    out[i] = v
                    break
        return out

    # ---- scalar R_unif_index(dn): one index in [0, dn), R's exact rbits rejection ----
    def _rbits(self, bits):
        """R's rbits(bits): draw (bits//16 + 1) 16-bit uniforms, keep low `bits` bits."""
        v = 0
        for _ in range(bits // 16 + 1):
            v = 65536 * v + int(math.floor(self.unif_rand() * 65536.0))
        return v & ((1 << bits) - 1)

    def _unif_index(self, dn):
        """R's R_unif_index(dn) -> integer in [0, dn). Honours sample_kind."""
        if dn <= 0:
            return 0
        if self.sample_kind == "rounding":
            return int(math.floor(dn * self.unif_rand()))
        bits = (int(dn) - 1).bit_length()         # ceil(log2 dn)
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    # ----------------- non-uniform variates (R nmath, bit-for-bit) ---------------
    def rexp(self, n, rate=1.0):
        """R's ``rexp(n, rate)`` -> float64 array (scale = 1/rate)."""
        scale = 1.0 / rate
        return np.array([_dist.rexp(self, scale) for _ in range(int(n))], dtype=np.float64)

    def rpois(self, n, mu):
        """R's ``rpois(n, mu)`` (scalar mu) -> int64 array of counts."""
        mu = float(mu)
        return np.array([_dist.rpois(self, mu) for _ in range(int(n))], dtype=np.int64)

    def rbinom(self, n, size, prob):
        """R's ``rbinom(n, size, prob)`` (scalar size, prob) -> int64 array of counts."""
        size = float(size)
        prob = float(prob)
        return np.array([_dist.rbinom(self, size, prob) for _ in range(int(n))], dtype=np.int64)

    def rgamma(self, n, shape, rate=1.0, scale=None):
        """R's ``rgamma(n, shape, rate=1, scale=1/rate)`` -> float64 array."""
        if scale is None:
            scale = 1.0 / rate
        shape = float(shape)
        scale = float(scale)
        return np.array([_dist.rgamma(self, shape, scale) for _ in range(int(n))], dtype=np.float64)

    # --------------------------- general R sample.int -----------------------------
    def sample(self, n, size=None, replace=False, prob=None):
        """R's ``sample.int(n, size, replace, prob)`` -> 0-based indices (int64 array).

        Matches R >= 3.6. Covers equal-prob (with/without replacement) and weighted
        ``prob=`` (with/without replacement, including R's Walker alias path). Defaults
        mirror R: ``size = n``, ``replace = False``, ``prob = None``.

        Note: R switches to a hashing algorithm only for ``n > 1e7`` non-replace
        unweighted draws (``sample.int(useHash=)``); that path is not replicated.
        """
        n = int(n)
        size = n if size is None else int(size)
        if prob is None:
            if replace:
                return self.sample_index(n, size)           # validated vectorized path
            return _sample.sample_noreplace(self, n, size)
        return _sample.sample_prob(self, n, size, replace, np.asarray(prob, dtype=np.float64))
