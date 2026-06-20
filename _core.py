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
import numpy as np

N, M = 624, 397
MATRIX_A, UPPER_MASK, LOWER_MASK = 0x9908b0df, 0x80000000, 0x7fffffff
MASK32 = 0xffffffff
I2_32M1 = 2.328306437080797e-10          # 1 / (2^32 - 1)
_LO = 0.5 * I2_32M1
_HI = 1.0 - 0.5 * I2_32M1

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
        u = raw * I2_32M1
        np.clip(u, _LO, _HI, out=u)              # R fixup: raw in [0, 2^32-1] -> u in (0,1)
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
        if bits > 16:
            # rbits draws ceil((bits+1)/16) uniforms per index; the vectorized fast
            # path below assumes a single 16-bit draw (n <= 2^16). Fall back to a
            # correct scalar loop for larger n (rare for bootstrap resampling).
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
