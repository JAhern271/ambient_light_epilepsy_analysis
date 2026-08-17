#!/usr/bin/env Rscript

cat("Loading R libaries...")

library(arrow)
library(data.table)

cat("Loading csv files...")

# ---- paths ----
raw_dir <- "/rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/data/PAXLUX_H/extracted"
out_dir <- "/rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/data/PAXLUX_H/parquet"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- list CSV files ----
csv_files <- list.files(
  raw_dir,
  pattern = "_Lux\\.csv$",
  full.names = TRUE
)

cat("Found", length(csv_files), "CSV files\n")

# ---- loop over participants ----
for (csv in csv_files) {

  # Extract SEQN from filename (e.g. 62161_Lux.csv)
  seqn <- sub("_Lux\\.csv$", "", basename(csv))
  out_file <- file.path(out_dir, paste0("SEQN_", seqn, ".parquet"))

  if (file.exists(out_file)) {
    cat("Skipping existing:", out_file, "\n")
    next
  }

  cat("Processing SEQN", seqn, "\n")

  # Fast CSV read
  dt <- fread(csv, showProgress = FALSE)

  # Optional: convert timestamp to POSIXct
  dt[, HEADER_TIMESTAMP := as.POSIXct(
    HEADER_TIMESTAMP,
    format = "%Y-%m-%d %H:%M:%S",
    tz = "UTC"
  )]

  # Write Parquet (compressed)
  write_parquet(
    dt,
    out_file,
    compression = "zstd"
  )

  rm(dt)
  gc()
}

cat("Done.\n")
