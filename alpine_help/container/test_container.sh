#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:05:00
#SBATCH --job-name=alpine_test
#SBATCH --output=alpine_test.%j.out
#SBATCH --partition=atesting
#SBATCH --account=ucb482_asc1

ml purge
export APPTAINER_BINDPATH=/scratch/local,/scratch/alpine,/projects,/curc,/pl/active
apptainer exec --no-mount /home --bind $PWD:/home/jovyan /projects/jogr4852/clotsimnet/alpine_help/container/fenics_stnb_gmsh_v4.sif python3 /projects/jogr4852/clotsimnet/data_gen/scripts/simulate.py /projects/jogr4852/clotsimnet/alpine_help/scripts/alpine_config.yaml
