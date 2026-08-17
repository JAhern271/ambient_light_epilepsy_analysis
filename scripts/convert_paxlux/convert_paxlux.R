#!/usr/bin/env Rscript
#
# Convert the per-participant light CSVs into 1 Hz parquet files.
#
#   Rscript convert_paxlux.R <cohort>
#
# e.g. Rscript convert_paxlux.R G
#
# The cohort used to be hard-coded, so the two cohorts were produced by
# different versions of this file. It is now an argument.
#
# Participants whose output already exists are skipped, so the job can be
# resubmitted after a timeout without redoing completed work.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript convert_paxlux.R <cohort>")
}

COHORT <- args[1]

.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.script_dir <- dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
source(file.path(dirname(.script_dir), "lib", "ale_paths.R"))

ale_check_cohort(COHORT)

raw_dir <- ale_lux_dir(COHORT, "extracted")
out_dir <- ale_lux_dir(COHORT, "parquet")

cat("Loading R libraries...\n")

library(arrow)
library(data.table)

ale_report_paths(COHORT)
cat("Input dir   :", raw_dir, "\n")
cat("Output dir  :", out_dir, "\n\n")

if (!dir.exists(raw_dir)) {
  stop("Input directory does not exist: ", raw_dir)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- list CSV files ----
csv_files <- list.files(
  raw_dir,
  pattern = "_Lux\\.csv$",
  full.names = TRUE
)

cat("Found", length(csv_files), "CSV files\n")
if (length(csv_files) == 0) {
  stop("No _Lux.csv files found in ", raw_dir)
}

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

cat("Done for cohort", COHORT, ".\n")
