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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript "${SCRIPT_DIR}/downsample_lux.R" "${COHORT}" "$@"
