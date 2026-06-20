# Generate golden RNG fixtures from REAL R, committed as JSON for the pure-Python
# rrng test suite to diff against. Run with:
#   Rscript rrng/tests/generate_golden.R
# Covers runif() and sample(replace=TRUE) under BOTH sample.kind methods across many
# seeds and n/size, the bits=15/16/17 boundary of R's rbits(), buffer-refill sizes,
# and THREADED streams (interleaved runif/sample from one set.seed, the key feature).
# No external packages -- base R only.

seeds <- c(1L, 42L, 100L, 2024L, 123456L)
# n values, chosen to span every code path:
#   .. 1000           : vectorized fast path (bits <= 15, single 16-bit rbits draw)
#   32768  (bits 15)  : last n on the fast path
#   32769, 40000, 65536 (bits 16) : rbits draws TWO 16-bit uniforms -> scalar path
#   65537, 100000      (bits 17)  : scalar path, 2 draws
ns    <- c(2L, 3L, 7L, 10L, 50L, 100L, 1000L,
           32768L, 32769L, 40000L, 65536L, 65537L, 100000L)
size  <- 20L
kinds <- c("Rejection", "Rounding")

num <- function(v) paste(format(v, digits = 17, scientific = FALSE, trim = TRUE), collapse = ", ")
int <- function(v) paste(as.integer(v), collapse = ", ")

cases <- character(0)

# --- runif blocks; include n past the 8192 internal buffer to test refill ---
for (s in seeds) {
  for (nn in c(25L, 10000L)) {
    set.seed(s)                       # runif is unaffected by sample.kind
    u <- runif(nn)
    cases <- c(cases, sprintf(
      '{"type": "runif", "seed": %d, "n": %d, "expected": [%s]}', s, nn, num(u)))
  }
}

# --- sample(seq_len(n), size, replace=TRUE) under both methods, 0-based ---
for (kind in kinds) {
  for (s in seeds) {
    for (n in ns) {
      suppressWarnings(RNGkind(sample.kind = kind))
      set.seed(s)
      idx <- sample(seq_len(n), size, replace = TRUE) - 1L   # to 0-based
      cases <- c(cases, sprintf(
        '{"type": "sample", "sample_kind": "%s", "seed": %d, "n": %d, "size": %d, "expected": [%s]}',
        tolower(kind), s, n, size, int(idx)))
    }
  }
}
suppressWarnings(RNGkind(sample.kind = "Rejection"))   # back to R >= 3.6 default

# --- large-size sample to exercise the internal buffer refill in sample_index ---
for (s in c(1L, 100L)) {
  set.seed(s)
  idx <- sample(seq_len(50L), 20000L, replace = TRUE) - 1L
  cases <- c(cases, sprintf(
    '{"type": "sample", "sample_kind": "rejection", "seed": %d, "n": 50, "size": 20000, "expected": [%s]}',
    s, int(idx)))
}

# --- THREADED stream: one set.seed(), interleaved runif/sample consumed in order.
# This is the library's core promise -- order & amount of consumption must match R. ---
for (s in seeds) {
  set.seed(s)                                    # default kind = Rejection (R >= 3.6)
  o1 <- runif(3)
  o2 <- sample(seq_len(50L),    5L, replace = TRUE) - 1L
  o3 <- runif(2)
  o4 <- sample(seq_len(40000L), 4L, replace = TRUE) - 1L   # bits=16, exercises scalar path mid-stream
  o5 <- runif(1)
  ops <- paste(
    sprintf('{"op": "runif", "n": 3, "expected": [%s]}', num(o1)),
    sprintf('{"op": "sample", "n": 50, "size": 5, "expected": [%s]}', int(o2)),
    sprintf('{"op": "runif", "n": 2, "expected": [%s]}', num(o3)),
    sprintf('{"op": "sample", "n": 40000, "size": 4, "expected": [%s]}', int(o4)),
    sprintf('{"op": "runif", "n": 1, "expected": [%s]}', num(o5)),
    sep = ", ")
  cases <- c(cases, sprintf(
    '{"type": "stream", "sample_kind": "rejection", "seed": %d, "ops": [%s]}', s, ops))
}

# robust output path regardless of cwd: write next to this script's fixtures dir
args <- commandArgs(trailingOnly = FALSE)
self <- sub("^--file=", "", args[grep("^--file=", args)])
base <- if (length(self)) dirname(normalizePath(self)) else "rrng/tests"
outdir <- file.path(base, "fixtures")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
out <- file.path(outdir, "golden_vectors.json")

writeLines(c("[", paste0("  ", cases, collapse = ",\n"), "]"), out)
cat("wrote", length(cases), "golden cases to", out, "\n")
cat("R version:", R.version.string, "\n")
