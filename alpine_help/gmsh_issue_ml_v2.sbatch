#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:05:00
#SBATCH --job-name=gmsh_issue_ml
#SBATCH --output=gmsh_issue_ml.%j.out
#SBATCH --partition=atesting
#SBATCH --account=ucb482_asc1

ml purge
ml intel/2022.1.2
ml impi/2021.5.0

export APPTAINER_BINDPATH=/scratch/local,/scratch/alpine,/projects,/curc,/pl/active

mpirun -np 1 apptainer exec --bind /scratch/alpine --bind /projects/jogr4852/FLATiron/include:/opt/conda/include /projects/jogr4852/clotsimnet/test_feature_extraction/scripts/container/fenics_stnb_gmsh.sif python3 simulate.py /projects/jogr4852/clotsimnet/test_feature_extraction/scripts/alpine_config.yaml
