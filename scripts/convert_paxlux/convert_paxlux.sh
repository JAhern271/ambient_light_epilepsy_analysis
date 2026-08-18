#!/bin/bash
#SBATCH --job-name=paxlux_to_parquet
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=convert_paxlux_%j.out
#SBATCH --error=convert_paxlux_%j.err
#
# Convert per-participant light CSVs to 1 Hz parquet.
#
#   sbatch convert_paxlux.sh G

set -euo pipefail

COHORT="${1:?Usage: sbatch convert_paxlux.sh <cohort>}"

# Slurm copies this script into a spool directory before running it, so
# BASH_SOURCE points somewhere useless under sbatch. SLURM_SUBMIT_DIR is the
# directory the job was submitted from, which is the repository root.
PROJECT_ROOT="${ALE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
export ALE_PROJECT_ROOT="$PROJECT_ROOT"

R_SCRIPT="${PROJECT_ROOT}/scripts/convert_paxlux/convert_paxlux.R"
if [[ ! -f "${R_SCRIPT}" ]]; then
    echo "Cannot find ${R_SCRIPT}" >&2
    echo "Submit from the repository root:" >&2
    echo "    cd <repo> && sbatch scripts/convert_paxlux/convert_paxlux.sh <cohort>" >&2
    echo "Or set ALE_PROJECT_ROOT to the repository root." >&2
    exit 1
fi

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript "${R_SCRIPT}" "${COHORT}"
