#!/usr/bin/env python3
import os
import sys
import time
import traceback
import argparse
# from simulate import file_cleanup
from flow_nse import RunFlow
import process_voxels

def find_mesh_h5(
    case_dir: str
) -> str:
    """
    Returns the path to the only .h5 mesh file in the given case directory.
    
    Parameters:
        case_dir (str): Path to the simulation case directory.
        
    Returns:
        str: Full path to the .h5 file.
        
    Raises:
        FileNotFoundError: If no .h5 file is found in the directory.
    """
    for fname in os.listdir(case_dir):
        if fname.endswith(".h5"):
            return os.path.join(case_dir, fname)
    raise FileNotFoundError(f"No .h5 file found in {case_dir}")

def file_cleanup(
    case_dir: str
):
    """
    Deletes unneeded files in the simulation directory to conserve disk space.
    
    Keeps only:
    - c_{pname}_000000.vtu files (where pname is a number)
    - c.pvd files
    - c000000.pvtu files
    - .vti files
    - .csv files

    Parameters:
        case_dir (str): Directory to simulation case.
        
    Returns:
        None
    """
    
    for root, _, files in os.walk(case_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.h5') or file.startswith('c'):
                print(f'Keeping {os.path.join(root, file)}')
            else:
                print(f'Removing {os.path.join(root, file)}')
                os.remove(os.path.join(root, file))
            
    print("File cleanup complete.")

def cfd_all(
    case_dir: str
):
    """
    Runs the full CFD pipeline for a single simulation case.
    
    Steps:
        1. Finds the mesh file (.h5)
        2. Runs Navier-Stokes CFD using the RunFlow class
        3 (Optional) Converts simulation output to VTI format
        4. Cleans up unneded files
        
    Parameters:
        case_dir (str): Path to a single simulation case directory.
        
    Returns:
        None
        
    Raises:
        Any exception raised during CFD or file cleanup will propagate here as well.
    """
    D = 0.0001 # Diffusivity
    max_vel = 20 # Max velocity
    sim_id = os.path.basename(case_dir)

    # 1) find the mesh file
    h5_path = find_mesh_h5(case_dir)
    print(f"[{sim_id}] Mesh: {h5_path}")

    # 2) run CFD
    start = time.time()
    print(f"[{sim_id}] Starting CFD")
    rf = RunFlow(case_dir=case_dir, D=D, max_vel=max_vel, hdf5_path=h5_path)
    rf.run_cfd()
    elapsed = time.time() - start
    print(f"[{sim_id}] CFD completed in {elapsed:.1f}s")

    # 3) (Optional) post‐processing—uncomment if needed
    # try:
    #     vti_file = os.path.join(case_dir, 'output', sim_id + '.vti')
    #     process_voxels.process_vti(vti_path=vti_file, case_dir=case_dir)
    #     print(f"[{sim_id}] VTI processing succeeded")
    # except Exception as e:
    #     print(f"[{sim_id}] VTI processing failed: {e}")

    # 4) cleanup
    try:
        file_cleanup(case_dir)
        print(f"[{sim_id}] Cleanup succeeded")
    except Exception as e:
        print(f"[{sim_id}] Cleanup failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run one CFD case; exit 0 on success, non-zero on error"
    )
    parser.add_argument(
        'case_dir',
        help="Path to a single bcc_lattice_* directory"
    )
    args = parser.parse_args()

    case = args.case_dir
    if not os.path.isdir(case):
        print(f"ERROR: '{case}' is not a directory", file=sys.stderr)
        sys.exit(2)

    try:
        cfd_all(case)
    except Exception:
        sim_id = os.path.basename(case)
        print(f"[{sim_id}] FATAL error:\n" + traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
