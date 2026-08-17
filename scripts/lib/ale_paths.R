# Path resolution for the R preprocessing scripts.
#
# The R equivalent of ambient_light_epilepsy/paths.py. Nothing in scripts/
# should contain an absolute path: source this and ask for the directory.
#
# The data root is resolved in this order:
#   1. the ALE_DATA_ROOT environment variable
#   2. <project root>/data
#   3. <project root>/../data   (the layout on the W: drive, where the
#      repository is checked out beside the data rather than above it)
#
# The project root comes from ALE_PROJECT_ROOT if the submission script
# exported it, otherwise it is derived from this script's own location.


ale_project_root <- function() {
  from_env <- Sys.getenv("ALE_PROJECT_ROOT")
  if (nzchar(from_env)) {
    return(normalizePath(from_env, mustWork = TRUE))
  }

  # Derive from the running script: <root>/scripts/<step>/<script>.R
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) {
    stop("Cannot determine the project root. Set ALE_PROJECT_ROOT, or run ",
         "this script with Rscript rather than interactively.")
  }

  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)
  normalizePath(dirname(dirname(dirname(script_path))), mustWork = TRUE)
}


ale_data_root <- function() {
  from_env <- Sys.getenv("ALE_DATA_ROOT")
  if (nzchar(from_env)) {
    return(normalizePath(from_env, mustWork = TRUE))
  }

  root <- ale_project_root()
  candidates <- c(file.path(root, "data"), file.path(dirname(root), "data"))

  for (candidate in candidates) {
    if (dir.exists(candidate)) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }

  stop("Could not find a data directory. Tried:\n  ",
       paste(candidates, collapse = "\n  "),
       "\nSet ALE_DATA_ROOT to the data directory.")
}


# Directory holding the raw NHANES tables for a cohort, e.g. data/G
ale_cohort_dir <- function(cohort) {
  file.path(ale_data_root(), cohort)
}


# Directories for the light recordings at each processing stage.
# Stage is "extracted", "parquet", or a downsampled "parquet_<n>min".
ale_lux_dir <- function(cohort, stage) {
  valid <- stage %in% c("extracted", "parquet") ||
    grepl("^parquet_[0-9.]+min$", stage)
  if (!valid) {
    stop("Unknown LUX stage '", stage, "'. Expected 'extracted', 'parquet', ",
         "or 'parquet_<n>min'.")
  }
  file.path(ale_data_root(), paste0("PAXLUX_", cohort), stage)
}


# Validate a cohort code supplied on the command line
ale_check_cohort <- function(cohort) {
  if (is.na(cohort) || !nzchar(cohort)) {
    stop("No cohort given. Expected a cohort code such as G or H.")
  }
  if (!cohort %in% c("G", "H")) {
    warning("Unrecognised cohort '", cohort, "'. Expected G (2011-12) or H (2013-14).")
  }
  # Invisible so that calling this at top level does not auto-print the cohort
  invisible(cohort)
}


# Print what was resolved, so a job log records where the data came from
ale_report_paths <- function(cohort) {
  cat("Project root:", ale_project_root(), "\n")
  cat("Data root   :", ale_data_root(), "\n")
  cat("Cohort      :", cohort, "\n")
}
