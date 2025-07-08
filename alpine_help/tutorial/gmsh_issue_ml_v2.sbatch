#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:05:00
#SBATCH --job-name=gmsh_issue_ml
#SBATCH --output=gmsh_issue_ml.%j.out
#SBATCH --partition=atesting
#SBATCH --account=ucb482_asc1

ml purge
ml gmsh/4.11.1

gmsh -nt $SLURM_NTASKS t1.geo -2
