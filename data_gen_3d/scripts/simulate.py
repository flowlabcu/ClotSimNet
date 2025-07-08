import pandas as pd
import time
import os
import shutil
from flow_nse import RunFlow
import geo_to_bc_dict as bc_dict
import subprocess
import yaml
import sys
import multiprocessing
import crop_image_dynamic
import process_voxels
from mesh import MeshGenerator
import textwrap
import stat
import subprocess
import pyvista as pv
import re

def file_cleanup(case_dir: str):
    '''
    Deletes unneeded files to keep dataset size as small as possible.
    Keeps only:
    - c_{pname}_000000.vtu files (where pname is a number)
    - c.pvd files
    - c000000.pvtu files
    - .vti files
    - .csv files

    Parameters:
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    '''
    
    # Regular expression pattern for matching c_{pname}_000000.vtu files
    vtu_pattern = re.compile(r'^c_\d+_000000\.vtu$')

    for root, _, files in os.walk(case_dir):
        for file in files:
            # Keep files only if they start with 'c' AND end with '.csv'
            if not (file.startswith('c') and file.endswith('.csv')):
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):  # Only remove if it's a file
                    os.remove(filepath)
                    print(f"Removed: {filepath}")
            
    print("File cleanup complete.")
    
def run_sim(D: float, max_vel: int, case_dir: str, hdf5_path: str):
    '''
    Run the CFD simulation
    
    Parameters:
        D (float): Diffusivity specified from YAML input file
        max_vel (int): Centerline/maximum flow velocity from YAML file
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    '''

    print(f'run_sim case dir: {case_dir}')
    
    run_flow = RunFlow(
        case_dir=case_dir,
        D=D,
        max_vel=max_vel,
        hdf5_path=hdf5_path
    )
    # Run CFD
    run_flow.run_cfd()

def run_simulation_from_csv(
    csv_path: str, 
    mesh_size: float, 
    D: float, 
    vel: float, 
    data_dir: str,
    row_index: int = 0):
    """
    Read parameters from CSV and run simulation for specified row.
    
    Parameters:
        csv_path (str): Path to the CSV file with tasks.
        mesh_size (float): Mesh size from input YAML file.
        D (float): Diffusivity from input YAML file.
        vel (float): Velocity from input YAML file.
        sources_dir (str): Path to 'sources' directory containing gmshUtilities.py and randSeqAdd.py.
        data_dir (str): Base directory where data is stored (used to construct output paths).
        row_index (int): Which row to process.
        
    Returns:
        None
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the specified row
    row = df.iloc[row_index]
    
    # Extract variables from row using column names
    index = row['index']
    case_dir = row['case_dir']
    sim_id = row['sim_id']
    radius = row['r_p']
    
    max_vel = vel
    
    start = time.time()
    
    # Calculate Peclet number
    peclet_number = (2* radius * max_vel) / D
    print(f'Peclet number: {peclet_number}')
    
    domain = (1.0, 1.0, 1.0)
    
    # Call meshing function
    mesh = MeshGenerator(
        radius=radius,
        mesh_size=mesh_size,
        domain=domain,
        case_dir=case_dir
    )
    h5_file_path = mesh.write_mesh()
    
    print('Starting CFD')
    
    # Call CFD function
    run_sim(D=D, max_vel=max_vel, case_dir=case_dir, hdf5_path=h5_file_path)
    print(f'CFD completed at {case_dir}')
    
    
    script_path = os.path.join(os.path.dirname(__file__), 'paraview_convert_vti.py')
    print(f'Script path: {script_path}')
    command = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch'

    print(f"Running ParaView conversion for case: {case_dir}")
    result = subprocess.run(
        [command, script_path, '--case-dir', case_dir],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ParaView conversion failed with error: {result.stderr}")
    else:
        print("ParaView conversion completed successfully")
    
    vti_path = os.path.join(case_dir, 'output', os.path.basename(case_dir) + '.vti')
    print(f'VTI path: {vti_path}')
    
    # Process VTI and append data to CSV
    process_voxels.process_vti(vti_path=vti_path, case_dir=case_dir)
    
    # Delete unnecessary files to keep space small
    file_cleanup(case_dir=case_dir)
    
def simulation_task_wrapper(row_index, csv_path, mesh_size, D, vel, data_dir):
    """
    Wrapper function to run a single simulation task.
    """
    try:
        run_simulation_from_csv(csv_path, mesh_size, D, vel, data_dir, row_index)
    except Exception as e:
        print(f"Simulation for row {row_index} encountered an error: {e}")

def main():
    try:
        # Configuration YAML file must be set as first argument when calling
        config_file = sys.argv[1]
    except IndexError:
        print("Error: Configuration YAML file must be specified, e.g.: python3 simulate.py /path/to/config_file.yaml")
        sys.exit(1)
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    ### ---- Pull out CFD parameters from YAML configuration file ---- ###
    mesh_size = config['mesh_size']
    D = config['diffusivity']
    vel = config['inlet_velocity']
    
    # Read in data directory and tasks file name from YAML.
    data_dir = config['data_dir']
    tasks_file_name = config['tasks_file_name']
    tasks_file = os.path.join(data_dir, tasks_file_name + '.csv')
    df = pd.read_csv(tasks_file)
    target_size = 512
    timeout_seconds = 900
    
    for row_index in range(len(df)):
    
        print(f"\nProcessing simulation {row_index + 1} of {len(df)} \t{round(((row_index + 1) / len(df)) * 100, 1)}%")
        
        simulation_task_wrapper(row_index=row_index, csv_path=tasks_file, mesh_size=mesh_size, D=D, vel=vel, data_dir=data_dir)
            
        print(f"Finished processing task {row_index}. Moving on to next simulation.")
        
if __name__ == '__main__':
    main()
