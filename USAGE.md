# Using `rrng`

A practical guide to reproducing R's default RNG from Python. If you only need the elevator
pitch, see the [README](README.md); this document is the how-to.

## Contents

- [Install & import](#install--import)
- [The mental model](#the-mental-model)
- [API reference](#api-reference)
- [R → Python cheat-sheet](#r--python-cheat-sheet)
- [Worked examples](#worked-examples)
  - [1. Match `runif`](#1-match-runif)
  - [2. Match `sample` (with replacement)](#2-match-sample-with-replacement)
  - [3. Bootstrap resampling a data array](#3-bootstrap-resampling-a-data-array)
  - [4. One seed threaded across a loop / `map()`](#4-one-seed-threaded-across-a-loop--map)
  - [5. Old R (< 3.6) scripts](#5-old-r--36-scripts)
- [Common pitfalls](#common-pitfalls)
- [Validating your own port](#validating-your-own-port)
- [What is and isn't supported](#what-is-and-isnt-supported)

---

## Install & import

```bash
pip install rrng            # from PyPI (if published)
pip install -e .            # or an editable install from a checkout
```

No install is strictly required: if the `rrng/` package directory is importable (on `PYTHONPATH`,
or sitting next to your script), this works directly:

```python
from rrng import RRNG
```

The only runtime dependency is NumPy.

## The mental model

R's randomness is a **single global stream**. `set.seed(s)` resets that stream; every subsequent
`runif`, `sample`, etc. consumes draws from it *in order*. To reproduce an R script you must:

1. Seed once, the same way (`RRNG(s)` ≙ `set.seed(s)`).
2. Consume draws in the **same order and the same amounts** as the R script.

`RRNG` is a stateful object that models exactly this one stream. Reuse the *same* `RRNG` instance
for the whole script; create a new one only where the R code calls `set.seed` again.

```python
g = RRNG(100)     # set.seed(100)
a = g.runif(5)    # first 5 draws
b = g.runif(5)    # next 5 draws  (NOT the same as a — the stream advanced)
```

## API reference

### `RRNG(seed, sample_kind="rejection")`
Create a stream seeded as R's `set.seed(seed)`.
- `seed` (int): the `set.seed()` argument.
- `sample_kind` (`"rejection"` | `"rounding"`): index algorithm for `sample_index`.
  `"rejection"` matches **R ≥ 3.6** (the current default); `"rounding"` matches **R < 3.6**.

### `g.set_seed(seed)`
Re-seed the existing object in place (≙ calling `set.seed(seed)` again). Returns `self`.

### `g.unif_rand() -> float`
One `runif(1)` draw.

### `g.runif(n) -> np.ndarray[float64]`
A length-`n` block, identical to R's `runif(n)`.

### `g.sample_index(n, size) -> np.ndarray[int64]`
The result of R's `sample(seq_len(n), size, replace = TRUE)`, returned as **0-based** indices
(so you can index NumPy arrays directly). Add `1` to compare with R's printed 1-based values.

### `g.sample(n, size=None, replace=False, prob=None) -> np.ndarray[int64]`
The general R `sample.int(n, size, replace, prob)` (defaults mirror R: `size=n`, `replace=False`,
`prob=None`). Covers equal-probability and weighted sampling, with and without replacement.
0-based indices.

### Variate generators
- `g.rnorm(n, mean=0.0, sd=1.0) -> float64` — R's `rnorm` (Inversion).
- `g.rexp(n, rate=1.0) -> float64` — R's `rexp`.
- `g.rpois(n, mu) -> int64` — R's `rpois` (scalar `mu`).
- `g.rbinom(n, size, prob) -> int64` — R's `rbinom` (scalar `size`, `prob`).
- `g.rgamma(n, shape, rate=1.0, scale=None) -> float64` — R's `rgamma` (`scale=1/rate`).

## R → Python cheat-sheet

| R | rrng |
|---|---|
| `set.seed(s)` | `g = RRNG(s)` |
| `runif(1)` | `g.unif_rand()` |
| `runif(n)` | `g.runif(n)` |
| `rnorm(n, mean, sd)` | `g.rnorm(n, mean, sd)` |
| `rexp(n, rate)` | `g.rexp(n, rate)` |
| `rpois(n, mu)` | `g.rpois(n, mu)` |
| `rbinom(n, size, prob)` | `g.rbinom(n, size, prob)` |
| `rgamma(n, shape, rate)` / `rgamma(n, shape, scale=)` | `g.rgamma(n, shape, rate=)` / `g.rgamma(n, shape, scale=)` |
| `sample(1:n, k, replace=TRUE)` | `g.sample_index(n, k) + 1` or `g.sample(n, k, replace=True) + 1` |
| `sample(1:n, k)` (no replacement) | `g.sample(n, k) + 1` |
| `sample(1:n, k, replace=TRUE, prob=w)` | `g.sample(n, k, replace=True, prob=w) + 1` |
| `x[sample(seq_along(x), length(x), replace=TRUE)]` | `x[g.sample_index(x.size, x.size)]` |
| `RNGkind(sample.kind="Rounding"); set.seed(s)` | `g = RRNG(s, sample_kind="rounding")` |

All `g.r*(n, ...)` methods return NumPy arrays; `rpois`/`rbinom` return `int64`, the rest `float64`.
`sample`/`sample_index` return **0-based** indices (add 1 to match R's printed values).

## Worked examples

### 1. Match `runif`

R:
```r
set.seed(100)
runif(5)
# 0.3077661 0.2576725 0.5523224 0.0563832 0.4685493
```

Python (full float64 precision; agrees with R to floating-point epsilon):
```python
from rrng import RRNG
g = RRNG(100)
print(g.runif(5))
# [0.30776611 0.2576725  0.55232243 0.05638315 0.46854928]
```

### 2. Match `sample` (with replacement)

R (≥ 3.6):
```r
set.seed(100)
sample(1:10, 10, replace = TRUE)
# 10 7 6 3 9 10 7 6 6 4
```

Python — `sample_index` returns 0-based, so add 1 to reproduce R's printed 1-based values:
```python
g = RRNG(100)
print(g.sample_index(10, 10))       # [9 6 5 2 8 9 6 5 5 3]   (0-based, index NumPy directly)
print(g.sample_index(10, 10) + 1)   # wrong here — the line above already advanced the stream!
```

⚠️ Each draw **advances the shared stream**, so calling `sample_index` twice gives two *different*
draws. To both index data and print R's 1-based values from the *same* draw, capture it once:

```python
g = RRNG(100)
idx = g.sample_index(10, 10)        # [9 6 5 2 8 9 6 5 5 3]   0-based
print(idx + 1)                      # [10  7  6  3  9 10  7  6  6  4]   == R's output
```

### 3. Bootstrap resampling a data array

R:
```r
set.seed(1)
B <- 5000
boot_means <- replicate(B, mean(sample(x, length(x), replace = TRUE)))
```

Python (same draws, same order):
```python
import numpy as np
from rrng import RRNG

g = RRNG(1)
B = 5000
boot_means = np.empty(B)
for b in range(B):
    resample = x[g.sample_index(x.size, x.size)]   # 0-based indices: index x directly
    boot_means[b] = resample.mean()
```

### 4. One seed threaded across a loop / `map()`

This is the case naive ports get wrong. R seeds **once**, then a loop/`map()` keeps drawing from
the same advancing stream:

```r
set.seed(100)
for (region in regions) {
  cf2 <- sample(cf[[region]], replace = TRUE)   # consumes draws...
  fa2 <- sample(fa[[region]], replace = TRUE)   # ...then more, same stream
  ...
}
```

Reproduce it with **one** `RRNG`, reused across iterations and in the same call order:

```python
g = RRNG(100)                       # seed ONCE, outside the loop
for region in regions:
    cf2 = cf[region][g.sample_index(cf[region].size, cf[region].size)]
    fa2 = fa[region][g.sample_index(fa[region].size, fa[region].size)]
    ...
```

Creating a fresh `RRNG(100)` inside the loop, or swapping the `cf`/`fa` order, would diverge.

### 5. Old R (< 3.6) scripts

Before R 3.6.0, `sample()` used the *Rounding* method. Reproduce those with `sample_kind="rounding"`:

```python
g = RRNG(42, sample_kind="rounding")
g.sample_index(100, 20)        # matches sample() under R < 3.6
```

`runif` is unaffected by `sample.kind`.

### 6. Non-uniform distributions

Every generator reproduces R's exact algorithm and consumes the stream identically, so they
compose in any order from one seed:

```r
set.seed(123)                          set.seed(123) equivalent:
x <- rnorm(1000, mean=5, sd=2)         g = RRNG(123)
y <- rexp(500, rate=0.5)               x = g.rnorm(1000, 5, 2)
k <- rpois(100, lambda=8)              y = g.rexp(500, rate=0.5)
b <- rbinom(100, size=20, prob=0.3)    k = g.rpois(100, 8)
gam <- rgamma(50, shape=2, rate=1.5)   b = g.rbinom(100, 20, 0.3)
                                       gam = g.rgamma(50, shape=2, rate=1.5)
```

`rnorm` uses R's default **Inversion** (two uniforms per value); `rpois`/`rbinom` pick R's
small- vs large-parameter algorithm automatically; `rgamma` uses GD (shape≥1) or GS (shape<1).
`rgamma` accepts either `rate=` or `scale=` (as in R; `scale = 1/rate`).

## Common pitfalls

- **0-based vs 1-based.** `sample_index` returns 0-based indices for direct NumPy indexing. Add `1`
  to compare against R's printed output.
- **Re-seeding inside a loop.** Build the `RRNG` once and reuse it; the stream must advance exactly
  as R's does. A new instance restarts the stream.
- **Call order.** If the R script does `sample(a); sample(b)`, your Python must call them in the
  same order — the second draw depends on how much the first consumed.
- **Wrong `sample_kind`.** Rejection (default) ≠ Rounding. A mismatch here is the classic "runif
  matches but sample doesn't" symptom.
- **Float formatting.** R prints 7 significant digits by default; `rrng` returns full-precision
  float64. They agree to within floating-point epsilon — compare numerically, not as strings.

## Validating your own port

Don't trust — verify. Generate golden vectors from your own R and diff:

```bash
Rscript rrng/tests/generate_golden.R     # writes tests/fixtures/golden_vectors.json
python  rrng/tests/test_rrng.py          # asserts rrng == R, bit-for-bit
```

For an end-to-end check, see `rrng/examples/` — it reproduces a published bootstrap attribution
table to every digit. The pattern there (dump R's input vectors once, then do all randomness in
Python) is the recommended way to port a real analysis.

## What is and isn't supported

**Supported (validated bit-for-bit):** Mersenne-Twister kind; `set.seed`; `runif`/`unif_rand`;
`rnorm` (Inversion); `rexp`; `rpois`; `rbinom`; `rgamma`; `sample`/`sample_index` equal- and
weighted-probability, with and without replacement (incl. Walker alias), under both Rejection
(R ≥ 3.6) and Rounding (R < 3.6).

**Not yet supported:** other continuous families (`rbeta`, `rchisq`, `rt`, `rf`, `rweibull`,
`rcauchy`, `rlogis`, …) and discrete (`rgeom`, `rnbinom`, `rhyper`); non-default RNG kinds and
`normal.kind`s; `sample.int(useHash=TRUE)` (only for `n > 1e7`). If your script uses these, the
covered parts still match, but the unsupported draws will diverge — and, because it's one shared
stream, divergence propagates to everything drawn afterward. Check the [roadmap](README.md#roadmap).
