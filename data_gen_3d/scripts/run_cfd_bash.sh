#!/usr/bin/env bash

# Directory containing all simulation case folders
DATA_DIR=/mnt/ssd1/joshgregory/clotsimnet_3d_300

# Maximum time allowed per simulation (in seconds)
TIMEOUT=3600

# Numper of MPI processes to use
NUM_RANKS=4

# Python script to run for each case
SCRIPT=/home/joshgregory/clotsimnet/data_gen_3d/scripts/cfd_single_with_bash.py

# Loop through all directories matching the name 'bcc_lattice*'
for case in "$DATA_DIR"/bcc_lattice*; do
  [[ -d "$case" ]] || continue
  echo "→ Processing $case"
  timeout --foreground $TIMEOUT \
    mpirun -np $NUM_RANKS python3 "$SCRIPT" "$case"
  rc=$?
  if   [[ $rc -eq 0  ]]; then echo "SUCCESS: $case"
  elif [[ $rc -eq 124 ]]; then echo "TIMEOUT: $case"
  else                         echo "FAILURE($rc): $case"
  fi
done
