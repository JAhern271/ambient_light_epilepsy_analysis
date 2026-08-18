#!/bin/bash
#SBATCH --job-name=xpt_to_parquet
#SBATCH --time=04:00:00
#SBATCH --mem=128G
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

# Slurm copies this script into a spool directory before running it, so
# BASH_SOURCE points somewhere useless under sbatch. SLURM_SUBMIT_DIR is the
# directory the job was submitted from, which is the repository root.
PROJECT_ROOT="${ALE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
export ALE_PROJECT_ROOT="$PROJECT_ROOT"

R_SCRIPT="${PROJECT_ROOT}/scripts/convert_xpt/convert_xpt.R"
if [[ ! -f "${R_SCRIPT}" ]]; then
    echo "Cannot find ${R_SCRIPT}" >&2
    echo "Submit from the repository root:" >&2
    echo "    cd <repo> && sbatch scripts/convert_xpt/convert_xpt.sh <cohort>" >&2
    echo "Or set ALE_PROJECT_ROOT to the repository root." >&2
    exit 1
fi

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript "${R_SCRIPT}" "${COHORT}" "$@"
