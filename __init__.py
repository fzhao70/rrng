"""rrng — reproduce R's native default RNG bit-for-bit in pure Python (no R at runtime).

    from rrng import RRNG
    g = RRNG(100)                       # R: set.seed(100)   (sample_kind="rounding" for R < 3.6)
    g.unif_rand(); g.runif(5)           # runif()
    g.rnorm(5); g.rexp(5, rate=2)       # rnorm (Inversion), rexp
    g.rpois(5, 3); g.rbinom(5, 20, .3)  # rpois, rbinom
    g.rgamma(5, shape=2, scale=2)       # rgamma
    g.sample(n, sz, replace=False, prob=w)   # general sample.int -> 0-based indices
    g.sample_index(n, sz)               # sample(seq_len(n), sz, replace=TRUE), 0-based

All of the above match R bit-for-bit (no R at runtime). See README.md / USAGE.md for scope,
the R->Python cheat-sheet, and the validation strategy.
"""
from ._core import RRNG

__all__ = ["RRNG"]
__version__ = "0.1.0"
