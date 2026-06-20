# Single-region (wus) attribution with set.seed(100); dump input vectors + result so the
# pure-Python R-RNG bootstrap can be validated against it. cwd = data-package-PNAS-BR.
suppressMessages({library(readr); library(dplyr); library(purrr); library(lubridate); library(extRemes)})
source("R/06_attribution_fun.R")
event_year <- 2026; event_month <- 3; event_date <- 15; event_range <- 10
event_year_range <- (event_year - event_range):(event_year + event_range)

files <- list.files("data/WUS-D3/", full = T, pattern = "csv")
wusd3 <- map_dfr(files, read_csv, col_types = "ccDd")
wus_swe <- wusd3 |> group_by(member, date) |> summarise(swe_km3 = sum(swe_km3), .groups="drop") |> mutate(huc2="wus")
wusd3 <- bind_rows(wus_swe, wusd3)
era5_swe <- read_csv("data/era5-land-swe.csv", show_col_types=FALSE) |> filter(huc2 != "wus") |> filter(huc2 %in% unique(wusd3$huc2))
era5_wus <- era5_swe |> group_by(date) |> summarise(swe_km3 = sum(swe_km3), .groups="drop") |> mutate(huc2="wus")
era5_swe <- bind_rows(era5_swe, era5_wus) |> filter(mday(date)==event_date) |>
  mutate(month=month(date), year=year(date)) |> group_by(month, huc2) |>
  mutate(mean_swe=mean(swe_km3), sd_swe=sd(swe_km3)) |> ungroup() |>
  mutate(z_score=(swe_km3-mean_swe)/sd_swe) |> filter(month==event_month)
wusd3_hist <- wusd3 |> filter(month(date)==event_month, mday(date)==event_date, year(date) %in% unique(era5_swe$year)) |>
  group_by(huc2) |> mutate(mean_swe=mean(swe_km3), sd_swe=sd(swe_km3)) |> ungroup()
stats <- wusd3_hist |> distinct(huc2, mean_swe, sd_swe)
wusd3_2 <- wusd3 |> filter(month(date)==event_month, mday(date)==event_date) |> left_join(stats, by="huc2") |>
  mutate(z_score=(swe_km3-mean_swe)/sd_swe)
era5_event <- era5_swe |> filter(year==event_year)

cf <- (wusd3_2 |> filter(huc2=="wus", year(date) %in% 1850:1900))$z_score
fa <- (wusd3_2 |> filter(huc2=="wus", year(date) %in% event_year_range))$z_score
ev <- (era5_event |> filter(huc2=="wus"))$z_score
writeLines(c(paste("n_cf", length(cf)), paste("n_fa", length(fa)), paste("event", ev)), "../analysis_source_compare/wus_vectors_meta.txt")
write_csv(data.frame(z=cf), "../analysis_source_compare/wus_cf.csv")
write_csv(data.frame(z=fa), "../analysis_source_compare/wus_fa.csv")

set.seed(100)
ans <- attribution_fun(counterfactual=cf, factual=fa, event=ev, alpha=0.05, N=5000)
p <- ans[[1]]
fac <- p[p$case=="factual",]; cnt <- p[p$case=="counterfactual",]; rr <- p[p$case=="event_RR",]
cat(sprintf("R seed=100 wus-only:  F %.3f [%.3f, %.3f]  CF %.2f [%.2f, %.2f]  RR %.4f [%.4f, %.4f]\n",
            1/fac$med, 1/fac$high, 1/fac$low, 1/cnt$med, 1/cnt$high, 1/cnt$low, rr$med, rr$low, rr$high))
