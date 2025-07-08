import pandas as pd
import time
import os
import glob
from flow_steady import RunFlow
from mesh_parallel_single import mesh
import geo_to_bc_dict as bc_dict
import tqdm
import subprocess
import yaml
import sys
import crop_image_dynamic
import process_images


def file_cleanup(case_dir):
    
    for root, _, files in os.walk(case_dir):
        for file in files:
            if not file.endswith(('.csv', '.jpeg')) and "crop" not in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                # print(f"Deleted: {file_path}")
            else:
                file_path = os.path.join(root, file)
                # print(f"Kept: {file_path}")
    # Remove empty "boudaries" directory
    boundaries_dir = os.path.join(case_dir, 'boundaries')
    if os.path.isdir(boundaries_dir) and not os.listdir(boundaries_dir):
        os.rmdir(boundaries_dir)
        # print(f'Removed directory: {boundaries_dir}')
    print("File cleanup complete.")
    
def run_sim(D, H, flow_rate, case_dir):
    bc_dict.write_bc_dict(geo_directory=case_dir)

    run_flow = RunFlow(
        case_dir=case_dir,
        D=D,
        H=H,
        flow_rate=flow_rate
    )

    # Run CFD
    run_flow.run_cfd()


def run_simulation_from_csv(csv_path, mesh_size, D, vel, sources_dir, row_index=0):
    """
    Read parameters from CSV and run simulation for specified row
    
    Args:
        csv_path (str): Path to the CSV file
        row_index (int): Which row to process
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the specified row
    row = df.iloc[row_index]
    
    # Extract variables from row using column names
    index = row['index']
    case_dir = row['case_dir']
    sim_id = row['sim_id']
    a_N = row['a_N']
    r_p = row['r_p']
    seed = row['seed']
    
    start = time.time()
    # num_cores = 1
    
    # Set inlet flow rate
    H = 1
    flow_rate = vel * H  # Velocity taken from IVISim paper
    
    # Calculate Peclet number
    peclet_number = (r_p*vel)/D
    print(f'Peclet number: {peclet_number}')
    
    # Call meshing function
    mesh(radius=r_p, 
         num_pores=a_N, 
         seed=seed, 
         mesh_size=mesh_size, 
         case_dir=case_dir,
         sources_dir=sources_dir)
    print(f'Meshing completed at {case_dir}')
    
    # Call CFD function
    run_sim(D=D, 
            H=H, 
            flow_rate=flow_rate, 
            case_dir=case_dir)
    print(f'CFD completed at {case_dir}')
    # print(f'Mesh/CFD completed in {time.time()-start} seconds')
    
    # Convert ParaView vtu to jpeg using Alpine's pvbatch installation
    
    # Flowlab2 pvbatch install path:
    command = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch' # For flowlab2
    # command = 'pvbatch' # For Alpine
    
    # result =subprocess.run([command, 'paraview_convert_alpine.py', case_dir], capture_output=True, text=True) # For Alpine
    
    result = subprocess.run([command, 'paraview_convert_flowlab2_local_pv.py', case_dir], capture_output=True, text=True) # For Flowlab2

    image_path = os.path.join(case_dir, 'output', os.path.basename(case_dir) + '.jpeg')
    print(f'Image path: {image_path}')
    
    # Crop the jpeg to remove whitespace
    cropped_image_path = crop_image_dynamic.remove_border_whitespace(uncropped_path=image_path)
    
    # Process image and append it to data.csv
    process_images.process_image(image_path=cropped_image_path, case_dir=case_dir)
    
    # Delete unnecessary files to keep space small
    file_cleanup(case_dir=case_dir)

# Loop through all rows
if __name__ == "__main__":
    
    try:
        # Configuration YAML file must be set as first argument when calling
        config_file = sys.argv[1]
    except IndexError:
        print("Error: Configuration YAML file must be specified, e.g.: python3 simulate.py /path/to/config_file.yaml")
        sys.exit(1)
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Pull out CFD parameters from YAML configuration file
    
    mesh_size = config['mesh_size']
    D = config['diffusivity']
    vel = config['inlet_velocity']
    
    # Read in sources directory
    sources_dir = config['sources_dir']
    
    # Read in tasks file
    data_dir = config['data_dir']
    tasks_file_name = config['tasks_file_name']
    tasks_file = os.path.join(data_dir, tasks_file_name + '.csv')
    df = pd.read_csv(tasks_file)
    
    
    
    for i in range(len(df)):
        print(f"\nProcessing simulation {i + 1} of {len(df)} \t{round(((i + 1) / len(df)) * 100, 1)}%")
        
        start = time.time()
        
        run_simulation_from_csv(csv_path=tasks_file, mesh_size=mesh_size, D=D, vel=vel, sources_dir=sources_dir, row_index=i)
        
        end = time.time()
        
        print(f'Run finished in {end-start} seconds')
