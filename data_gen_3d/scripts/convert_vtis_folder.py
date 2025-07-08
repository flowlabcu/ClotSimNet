import os
import sys
import argparse
import process_voxels
import subprocess
from tqdm import tqdm


def process_vti(
    output_dir: str
):
    """
    Runs ParaView post-processing and converts to .vti
    
    Parameters:
        output_dir (str): Path to the simulation case directory.
    """
    
    # --- Post-processing steps start here ---
    
    # Path to ParaView's pvbatch and the conversion script
    pvbatch_cmd = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch'
    conv_script = os.path.join(os.path.dirname(__file__), 'paraview_convert_vti.py')
    
    print("Running ParaView conversion")
    result = subprocess.run(
        [pvbatch_cmd, conv_script, '--case-dir', output_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ParaView conversion failed: {result.stderr}")
    else:
        print("ParaView conversion succeeded")

    # Path to resulting VTI file
    vti_path = os.path.join(output_dir, 'output', os.path.basename(output_dir) + '.vti')
    print(f"Processing VTI → {vti_path}")
    try:
        process_voxels.process_vti(vti_path=vti_path, case_dir=output_dir)
        print("VTI processing succeeded")
    except Exception as e:
        print(f"VTI processing failed: {e}")
        
def main():
    """
    Recursively searches for simulation case directories under a root directory and processes each one if it contains a 'c.pvd' file. Note that the corresponding 'c.vtu' files also must exist.
    """
    # CHANGE ROOT DIR AS NEEDED
    root_dir = '/mnt/ssd1/joshgregory/clotsimnet_3d_300'
    print(f'Root directory: {root_dir}')
    
    case_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'c.pvd':
                full_path = os.path.join(dirpath, filename)
                
                parts = full_path.split(os.sep)
                if 'output' in parts:
                    idx = parts.index('output')
                    case_dir_path = os.sep.join(parts[:idx])
                    case_dirs.append(case_dir_path)
                else:
                    print('No output directory')
                    
    for case_dir in tqdm(case_dirs, desc='Processing cases'):
        process_vti(case_dir)
    
if __name__=='__main__':
    main()
    
