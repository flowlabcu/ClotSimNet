#!/bin/bash

#SBATCH --partition=amilan
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --job=build
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL                # Email when job ends and/or fails
#SBATCH --mail-user=jogr4852@colorado.edu   # Email address
#SBATCH --output=build_v5.out       # Output file name

#This will go faster if you do it on the SSD on the node
cd $SLURM_SCRATCH
cp /projects/jogr4852/clotsimnet/alpine_help/container/build_fenics_stnb_gmsh_v5_local.def .

#build
apptainer build fenics_stnb_gmsh_v5.sif build_fenics_stnb_gmsh_v5_local.def

#now copy finished container back to projects directory
cp fenics_stnb_gmsh_v5.sif /projects/jogr4852/clotsimnet/alpine_help/container/

