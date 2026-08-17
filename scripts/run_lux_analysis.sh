#!/bin/bash
#SBATCH --job-name=lux_analysis
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=run_lux_analysis_%j.out
#SBATCH --error=run_lux_analysis_%j.err
#
# Compute the light metrics and join covariates.
#
#   sbatch run_lux_analysis.sh              # 5 minute data
#   sbatch run_lux_analysis.sh --downsample 1hz
#
# Arguments are passed straight through to scripts/lux_analysis.py; see
# `python scripts/lux_analysis.py --help`. Which dataset to run used to be
# chosen by editing a variable in the Python file.
#
# The virtual environment defaults to <project root>/venv. Override with
# ALE_VENV if yours lives elsewhere:
#
#   ALE_VENV=/path/to/venv sbatch run_lux_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALE_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

module purge
module load bluebear
module load bear-apps/2023a
module load Python

VENV="${ALE_VENV:-${ALE_PROJECT_ROOT}/venv}"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "No virtual environment at ${VENV}" >&2
    echo "Create one, or set ALE_VENV to its location." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

cd "${ALE_PROJECT_ROOT}"
python scripts/lux_analysis.py "$@"
