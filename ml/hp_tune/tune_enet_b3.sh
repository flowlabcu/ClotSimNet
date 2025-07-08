#!/bin/bash
#SBATCH --partition=gh200
#SBATCH --nodes=1                           # Number of requested nodes
#SBATCH --qos=gh200
#SBATCH --account=ucb620_asc1
#SBATCH --ntasks-per-node=72
#SBATCH --gres=gpu
#SBATCH --time=24:00:00                      # Max walltime in h:m:s
#SBATCH --nodelist=c3gh-c13-u26
#SBATCH --job-name=tune_enet_b3                # job name
#SBATCH --mail-type=BEGIN,END,FAIL                # Email when job ends and/or fails
#SBATCH --mail-user=jogr4852@colorado.edu   # Email address
#SBATCH --output=tune_enet_b3.out       # Output file name


# Written by:   Shelley Knuth, 24 February 2014
# Updated by:   Andrew Monaghan, 08 March 2018
# Updated by:   Kim Kanigel Winner, 23 June 2018
# Updated by:   Shelley Knuth, 17 May 2019
# Updated by:   Josh Gregory, 29 March 2024
# Purpose:      PyTorch job scripting template


module purge

source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh

mamba deactivate
mamba activate clotsimnet

python3 tune_enet_b3.py