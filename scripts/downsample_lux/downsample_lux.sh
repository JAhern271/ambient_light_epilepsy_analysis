#!/bin/bash
#SBATCH --job-name=lux_downsample
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=downsample_lux_%j.out
#SBATCH --error=downsample_lux_%j.err
#
# Bin 1 Hz light recordings to fixed-width means.
#
#   sbatch downsample_lux.sh G
#   sbatch downsample_lux.sh H 1 start
#
# Arguments are passed through to the R script: cohort, then optionally
# bin_minutes and time_align.

set -euo pipefail

COHORT="${1:?Usage: sbatch downsample_lux.sh <cohort> [bin_minutes] [time_align]}"
shift

# Slurm copies this script into a spool directory before running it, so
# BASH_SOURCE points somewhere useless under sbatch. SLURM_SUBMIT_DIR is the
# directory the job was submitted from, which is the repository root.
PROJECT_ROOT="${ALE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
export ALE_PROJECT_ROOT="$PROJECT_ROOT"

R_SCRIPT="${PROJECT_ROOT}/scripts/downsample_lux/downsample_lux.R"
if [[ ! -f "${R_SCRIPT}" ]]; then
    echo "Cannot find ${R_SCRIPT}" >&2
    echo "Submit from the repository root:" >&2
    echo "    cd <repo> && sbatch scripts/downsample_lux/downsample_lux.sh <cohort>" >&2
    echo "Or set ALE_PROJECT_ROOT to the repository root." >&2
    exit 1
fi

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript "${R_SCRIPT}" "${COHORT}" "$@"
