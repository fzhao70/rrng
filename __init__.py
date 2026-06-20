"""rrng — reproduce R's native default RNG bit-for-bit in pure Python (no R at runtime).

    from rrng import RRNG
    g = RRNG(100)                 # R: set.seed(100)   (sample_kind="rounding" for R < 3.6)
    g.unif_rand()                 # one R runif() draw
    g.runif(5)                    # an R runif(5) block (numpy array)
    idx = g.sample_index(n, sz)   # 0-based sample(seq_len(n), sz, replace=TRUE)

See README.md for scope and the validation strategy.
"""
from ._core import RRNG

__all__ = ["RRNG"]
__version__ = "0.1.0"
