#!/usr/bin/env Rscript
#
# Convert raw NHANES .xpt tables to parquet.
#
#   Rscript convert_xpt.R <cohort> [table ...]
#
# e.g. Rscript convert_xpt.R G                 # all standard tables
#      Rscript convert_xpt.R H PAXMIN          # just one
#
# The cohort and table used to be hard-coded to PAXMIN_H, so converting
# anything else meant editing this file and losing the previous version.
#
# Existing parquet files are skipped unless ALE_OVERWRITE=1, so a rerun after
# a timeout does not redo completed conversions.

DEFAULT_TABLES <- c("DEMO", "DEQ", "DPQ", "MCQ", "OCQ", "PAXHD", "PAXMIN", "RXQ_RX")

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript convert_xpt.R <cohort> [table ...]")
}

COHORT <- args[1]
TABLES <- if (length(args) >= 2) args[-1] else DEFAULT_TABLES
# Treat only a meaningful value as true, so ALE_OVERWRITE=0 does not enable it
OVERWRITE <- tolower(Sys.getenv("ALE_OVERWRITE")) %in% c("1", "true", "yes")

.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.script_dir <- dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
source(file.path(dirname(.script_dir), "lib", "ale_paths.R"))

ale_check_cohort(COHORT)

library(arrow)
library(haven)

cohort_dir <- ale_cohort_dir(COHORT)

ale_report_paths(COHORT)
cat("Cohort dir  :", cohort_dir, "\n")
cat("Tables      :", paste(TABLES, collapse = ", "), "\n")
cat("Overwrite   :", OVERWRITE, "\n\n")

if (!dir.exists(cohort_dir)) {
  stop("Cohort directory does not exist: ", cohort_dir)
}

converted <- 0
skipped <- 0
missing <- character(0)

for (table in TABLES) {

  stem <- paste0(table, "_", COHORT)
  xpt_path <- file.path(cohort_dir, paste0(stem, ".xpt"))
  parquet_path <- file.path(cohort_dir, paste0(stem, ".parquet"))

  if (!file.exists(xpt_path)) {
    cat("No XPT for", stem, "- skipping\n")
    missing <- c(missing, stem)
    next
  }

  if (file.exists(parquet_path) && !OVERWRITE) {
    cat("Already converted:", basename(parquet_path), "\n")
    skipped <- skipped + 1
    next
  }

  cat("Reading XPT from:", xpt_path, "\n")
  df <- read_xpt(xpt_path)
  cat("  rows:", nrow(df), " cols:", ncol(df), "\n")

  # Some NHANES string columns carry invalid encodings that break the
  # parquet writer, so force everything character to UTF-8 first.
  df[] <- lapply(df, function(col) {
    if (is.character(col)) enc2utf8(col) else col
  })

  cat("  writing:", parquet_path, "\n")
  write_parquet(df, parquet_path)
  converted <- converted + 1
}

cat("\nConverted:", converted, " skipped:", skipped,
    " missing:", length(missing), "\n")
if (length(missing) > 0) {
  cat("Missing XPT files:", paste(missing, collapse = ", "), "\n")
}
cat("Done for cohort", COHORT, ".\n")
