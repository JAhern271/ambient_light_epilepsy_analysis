#!/bin/bash
#SBATCH --job-name=xpt_to_parquet
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=convert_xpt_%j.out
#SBATCH --error=convert_xpt_%j.err
#
# Convert raw NHANES .xpt tables to parquet.
#
#   sbatch convert_xpt.sh G              # all standard tables
#   sbatch convert_xpt.sh H PAXMIN       # just one
#
# Set ALE_OVERWRITE=1 to replace parquet files that already exist.

set -euo pipefail

COHORT="${1:?Usage: sbatch convert_xpt.sh <cohort> [table ...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript "${SCRIPT_DIR}/convert_xpt.R" "${COHORT}" "$@"
