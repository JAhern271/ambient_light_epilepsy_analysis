#!/bin/bash
#SBATCH --job-name=lux_downsample
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=downsample_lux.out
#SBATCH --error=downsample_lux.err

module purge
module load bluebear
module load bear-apps/2024a
module load R/4.5.0-gfbf-2024a
module load arrow-R/17.0.0.1-foss-2024a-R-4.5.0

Rscript /rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/scripts/downsample_lux/downsample_lux.R
