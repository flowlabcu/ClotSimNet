#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:05:00
#SBATCH --job-name=gmsh_issue_ml
#SBATCH --output=gmsh_issue_ml.out
#SBATCH --partition=atesting
#SBATCH --account=ucb482_asc1
#SBATCH --mail-type=END,FAIL                # Email when job ends and/or fails
#SBATCH --mail-user=jogr4852@colorado.edu

ml purge
ml intel/2022.1.2
ml impi/2021.5.0
ml singularity/3.7.4
ml gmsh
export SLURM_EXPORT_ENV=ALL

mpirun -np 1 singularity exec --bind /scratch/alpine --bind /projects/jogr4852/FLATiron/include:/opt/conda/include --env CXX=mpic++ --env CC=mpicc --env FC=mpifort /projects/jogr4852/FLATiron/fenics_stnb.sif gmsh
