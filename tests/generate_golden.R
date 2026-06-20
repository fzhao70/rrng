# Generate golden RNG fixtures from REAL R, committed as JSON for the pure-Python
# rrng test suite to diff against. Run with:
#   /glade/u/apps/opt/miniforge/envs/r-4.5/bin/Rscript rrng/tests/generate_golden.R
# Covers runif() and sample(replace=TRUE) under BOTH sample.kind methods, several
# seeds and n/size. No external packages — base R only.

seeds <- c(1L, 42L, 100L, 2024L, 123456L)
ns    <- c(2L, 3L, 7L, 10L, 50L, 100L, 1000L, 100000L)  # 100000 -> bits=17, scalar rejection path
size  <- 20L
kinds <- c("Rejection", "Rounding")

esc <- function(s) gsub('"', '\\"', s, fixed = TRUE)
num <- function(v) paste(format(v, digits = 17, scientific = FALSE, trim = TRUE), collapse = ", ")
int <- function(v) paste(as.integer(v), collapse = ", ")

cases <- character(0)

# --- runif blocks (sample.kind is irrelevant to runif) ---
for (s in seeds) {
  set.seed(s)                       # default kind; runif unaffected by sample.kind
  u <- runif(25)
  cases <- c(cases, sprintf(
    '{"type": "runif", "seed": %d, "n": 25, "expected": [%s]}', s, num(u)))
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
