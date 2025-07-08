import sys
import os
import yaml
import pandas as pd
import subprocess
import time
import subprocess
import signal
from simulate import file_cleanup

from flow_nse import RunFlow
import process_voxels
# from simulate import file_cleanup   # assuming you’ve defined file_cleanup there

def find_mesh_h5(case_dir: str) -> str:
    """Return the only .h5 file in the given case directory."""
    for fname in os.listdir(case_dir):
        if fname.endswith(".h5"):
            return os.path.join(case_dir, fname)
    raise FileNotFoundError(f"No .h5 file found in {case_dir}")

def cfd_all(data_dir: str, case_dir):

    D           = 0.0001
    max_vel     = 20
    data_dir    = data_dir
    tasks_file_name = 'tasks'
    tasks_csv   = os.path.join(data_dir, tasks_file_name + '.csv')
    # sim_suffix  = cfg.get('sim_id_suffix', ".h5")

    # Path to your pvbatch executable and conversion script
    pvbatch_cmd = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch'
    conv_script = os.path.join(os.path.dirname(__file__), 'paraview_convert_vti.py')

    sim_id   = os.path.basename(case_dir)
    h5_path  = h5_path = find_mesh_h5(case_dir)
    print(h5_path)

    start = time.time()
    
    print(f"Starting CFD → {case_dir}")
    rf = RunFlow(case_dir=case_dir, D=D, max_vel=max_vel, hdf5_path=h5_path)
    rf.run_cfd()
    
    end = time.time()
    
    print(f"Completed CFD")
    print(f'Time taken: {end - start} seconds')
    
    # --- Post-processing steps start here ---
    # print(f"[{idx:3d}] Running ParaView conversion")
    # result = subprocess.run(
    #     [pvbatch_cmd, conv_script, '--case-dir', case_dir],
    #     capture_output=True, text=True
    # )
    # if result.returncode != 0:
    #     print(f"ParaView conversion failed: {result.stderr}")
    # else:
    #     print("ParaView conversion succeeded")

    # vti_path = os.path.join(case_dir, 'output', os.path.basename(case_dir) + '.vti')
    # print(f"[{idx:3d}] Processing VTI → {vti_path}")
    # try:
    #     process_voxels.process_vti(vti_path=vti_path, case_dir=case_dir)
    #     print("VTI processing succeeded")
    # except Exception as e:
    #     print(f"VTI processing failed: {e}")

    # print(f"[{idx:3d}] Cleaning up files in {case_dir}")
    # try:
    #     file_cleanup(case_dir)
    # except Exception as e:
    #     print(f"Cleanup error: {e}")

if __name__ == '__main__':
    data_dir = '/home/joshgregory/clotsimnet/data_gen_3d/lattice_test_10'

    # with open('run_all.sh', 'w') as f:
    #     for file in os.listdir(data_dir):
    #         if file.startswith('bcc_lattice'):
    #             case_dir = os.path.join(data_dir, file)
    #             # print(case_dir)

    #             f.write(f'mpirun -n 4 python3 {case_dir}\n')


                # cfd_all(data_dir, case_dir)
            #   case_dir = '/home/joshgregory/clotsimnet/data_gen_3d/lattice_test_20/bcc_lattice_rp_05872'


    for file in os.listdir(data_dir):
        if file.startswith('bcc_lattice'):
            case_dir = os.path.join(data_dir, file)
            # print(case_dir)


            print(f'Starting CFD at {case_dir}')
            
            cfd_all(data_dir, case_dir)
        #   case_dir = '/home/joshgregory/clotsimnet/data_gen_3d/lattice_test_20/bcc_lattice_rp_05872'
