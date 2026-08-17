#!/bin/bash
#SBATCH --job-name=lux_analysis
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=run_lux_analysis.out
#SBATCH --error=run_lux_analysis.err

module purge; module load bluebear
module load bear-apps/2023a
module load Python

# Activate your venv
source /rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis/ambient_light_epilepsy_analysis/venv/bin/activate

# Go to project root
#cd /rds/projects/t/terryjr-fellowship-ahern/projects/ambient_light_epilepsy_analysis

# Run script
python lux_analysis.py