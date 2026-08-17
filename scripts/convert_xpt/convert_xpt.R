cat("Working directory:", getwd(), "\n")

library(arrow)
library(haven)

# Get project root (one level up from scripts/)
project_root <- "/rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis"

# Define paths
xpt_path <- file.path(project_root, "data", "H", "PAXMIN_H.xpt")
parquet_path <- file.path(project_root, "data", "H", "PAXMIN_H.parquet")

cat("Reading XPT from:", xpt_path, "\n")
df <- read_xpt(xpt_path)

cat("Rows:", nrow(df), "\n")

df[] <- lapply(df, function(col) {
  if (is.character(col)) enc2utf8(col) else col
})

cat("Writing Parquet to:", parquet_path, "\n")

write_parquet(df, parquet_path)

cat("Done.\n")
