import sys
import os
import yaml
import pandas as pd
import subprocess
import time
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

def cfd_all(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    D           = cfg['diffusivity']
    max_vel     = cfg['inlet_velocity']
    data_dir    = cfg['data_dir']
    tasks_csv   = os.path.join(data_dir, cfg['tasks_file_name'] + '.csv')
    sim_suffix  = cfg.get('sim_id_suffix', ".h5")

    df = pd.read_csv(tasks_csv)

    # Path to your pvbatch executable and conversion script
    pvbatch_cmd = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch'
    conv_script = os.path.join(os.path.dirname(__file__), 'paraview_convert_vti.py')

    for idx, row in df.iterrows():
        case_dir = row['case_dir']
        sim_id   = row['sim_id']
        h5_path  = h5_path = find_mesh_h5(case_dir)
        print(h5_path)

        start = time.time()
        
        print(f"[{idx:3d}] Starting CFD → {case_dir}")
        rf = RunFlow(case_dir=case_dir, D=D, max_vel=max_vel, hdf5_path=h5_path)
        rf.run_cfd()
        
        end = time.time()
        
        print(f"[{idx:3d}] Completed CFD")
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

        print(f"[{idx:3d}] Done with case\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: mpirun -np <cores> python cfd_all.py /path/to/config.yaml")
        sys.exit(1)
    cfd_all(sys.argv[1])
