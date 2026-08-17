#!/bin/bash
#SBATCH --job-name=paxlux_to_parquet
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=convert_paxlux.out
#SBATCH --error=convert_paxlux.err

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0


Rscript /rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/scripts/convert_paxlux/convert_paxlux.R
