# ----------------------------
# Config (EDIT THIS)
# ----------------------------
BIN_MINUTES <- 5        # set to 1 for 1-min data
TIME_ALIGN  <- "center" # "center" or "start"

# ----------------------------
# Libraries
# ----------------------------
cat("Loading modules... \n")
library(arrow)
library(data.table)
library(lubridate)

# ----------------------------
# Paths
# ----------------------------
input_dir  <- "/rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/data/PAXLUX_H/parquet"
output_dir <- "/rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/data/PAXLUX_H/parquet_5min"

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
cat("Processing files...\n")
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

for (f in files) {
  process_file(f)
}

message("All files processed.")
