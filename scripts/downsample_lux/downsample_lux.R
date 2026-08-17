#!/usr/bin/env Rscript
#
# Bin 1 Hz light recordings down to fixed-width means.
#
#   Rscript downsample_lux.R <cohort> [bin_minutes] [time_align]
#
# e.g. Rscript downsample_lux.R G
#      Rscript downsample_lux.R H 1 start
#
# The cohort used to be hard-coded behind an "EDIT THIS" block, so the two
# cohorts were produced by different versions of this file and neither version
# was recorded. It is now an argument, and the settings used are echoed into
# the job log.

# ----------------------------
# Arguments
# ----------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript downsample_lux.R <cohort> [bin_minutes] [time_align]")
}

COHORT      <- args[1]
BIN_MINUTES <- if (length(args) >= 2) as.numeric(args[2]) else 5
TIME_ALIGN  <- if (length(args) >= 3) args[3] else "center"

if (!TIME_ALIGN %in% c("center", "start")) {
  stop("time_align must be 'center' or 'start', got '", TIME_ALIGN, "'")
}

# ----------------------------
# Paths
# ----------------------------
.file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
.script_dir <- dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
source(file.path(dirname(.script_dir), "lib", "ale_paths.R"))

ale_check_cohort(COHORT)

input_dir  <- ale_lux_dir(COHORT, "parquet")
output_dir <- ale_lux_dir(COHORT, paste0("parquet_", BIN_MINUTES, "min"))

# ----------------------------
# Libraries
# ----------------------------
cat("Loading modules... \n")
library(arrow)
library(data.table)
library(lubridate)

ale_report_paths(COHORT)
cat("Bin minutes :", BIN_MINUTES, "\n")
cat("Time align  :", TIME_ALIGN, "\n")
cat("Input dir   :", input_dir, "\n")
cat("Output dir  :", output_dir, "\n\n")

if (!dir.exists(input_dir)) {
  stop("Input directory does not exist: ", input_dir)
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------
# Helper: bin timestamp
# ----------------------------
bin_timestamp <- function(t, bin_minutes, align = "center") {

  # floor to bin start using POSIX arithmetic
  bin_seconds <- bin_minutes * 60
  t_num <- as.numeric(t)

  bin_start <- as.POSIXct(
    floor(t_num / bin_seconds) * bin_seconds,
    origin = "1970-01-01",
    tz = "UTC"
  )

  if (align == "start") {
    return(bin_start)
  }

  if (align == "center") {
    return(bin_start + bin_seconds / 2)
  }

  stop("Unknown TIME_ALIGN option")
}


# ----------------------------
# Process one file
# ----------------------------
process_file <- function(parquet_path) {

  message("Processing: ", parquet_path)

  # Read only required columns
  dt <- as.data.table(
    read_parquet(parquet_path, col_select = c("HEADER_TIMESTAMP", "LUX"))
  )

  # Parse timestamp
  dt[, HEADER_TIMESTAMP := as.POSIXct(
    HEADER_TIMESTAMP, tz = "UTC"
  )]

  # Create bin
  dt[, bin_time := bin_timestamp(
    HEADER_TIMESTAMP,
    bin_minutes = BIN_MINUTES,
    align = TIME_ALIGN
  )]

  # Aggregate
  out <- dt[
    ,
    .(
      mean_lux  = mean(LUX, na.rm = TRUE),
      n_samples = .N
    ),
    by = bin_time
  ]

  setnames(out, "bin_time", "timestamp")

  # Output filename
  seqn <- gsub("^SEQN_|\\.parquet$", "", basename(parquet_path))
  out_path <- file.path(
    output_dir,
    paste0("SEQN_", seqn, "_", BIN_MINUTES, "min.parquet")
  )

  write_parquet(out, out_path)
}

# ----------------------------
# Main
# ----------------------------
files <- list.files(
  input_dir,
  pattern = "^SEQN_.*\\.parquet$",
  full.names = TRUE
)

cat("Found", length(files), "input files\n")
if (length(files) == 0) {
  stop("No input files found in ", input_dir)
}

cat("Processing files...\n")
for (f in files) {
  process_file(f)
}

message("All files processed for cohort ", COHORT, ".")
