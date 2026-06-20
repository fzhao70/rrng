# Dump the paper's exact z-score vectors (cf, fa, event) per region in the 07 processing
# order, so the pure-Python RRNG bootstrap can reproduce the PUBLISHED values bit-for-bit.
# cwd = data-package-PNAS-BR.
suppressMessages({library(readr); library(dplyr); library(purrr); library(lubridate)})
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

outdir <- "../analysis_source_compare/paper_vectors"
dir.create(outdir, showWarnings = FALSE)
order <- unique(era5_swe$huc2)                 # the 07 processing order
writeLines(order, file.path(outdir, "region_order.txt"))
for (h in order) {
  cf <- (wusd3_2 |> filter(huc2==h, year(date) %in% 1850:1900))$z_score
  fa <- (wusd3_2 |> filter(huc2==h, year(date) %in% event_year_range))$z_score
  ev <- (era5_event |> filter(huc2==h))$z_score
  write_csv(data.frame(z=cf), file.path(outdir, paste0("cf_", h, ".csv")))
  write_csv(data.frame(z=fa), file.path(outdir, paste0("fa_", h, ".csv")))
  writeLines(as.character(ev), file.path(outdir, paste0("event_", h, ".txt")))
}
cat("region order:", paste(order, collapse=", "), "\n")
cat("dumped cf/fa/event vectors to", outdir, "\n")
