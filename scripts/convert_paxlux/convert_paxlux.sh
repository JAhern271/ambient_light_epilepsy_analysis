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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript "${SCRIPT_DIR}/convert_paxlux.R" "${COHORT}"
