import pandas as pd
import time
import signal
import datetime
import os
import shutil
from flow_steady import RunFlow
import geo_to_bc_dict as bc_dict
import subprocess
import yaml
import sys
import crop_image_dynamic
import process_images
from mesh import MeshGenerator

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

def file_cleanup(case_dir: str):
    '''
    Deletes unneeded files to keep dataset size as small as possible.
    
    Parameters:
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    '''
    
    for root, _, files in os.walk(case_dir):
        for file in files:
            if not file.endswith(('.csv', '.jpeg')) and 'cnn' not in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            else:
                file_path = os.path.join(root, file)
    
    # Remove empty "boudaries" directory
    boundaries_dir = os.path.join(case_dir, 'boundaries')
    if os.path.isdir(boundaries_dir) and not os.listdir(boundaries_dir):
        os.rmdir(boundaries_dir)
        
    # Remove non-empty 'output' directory
    output_dir = os.path.join(case_dir, 'output')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    print("File cleanup complete.")
    
def run_sim(D: float, H: int, flow_rate: int, case_dir: str):
    '''
    Run the CFD simulation
    
    Parameters:
        D (float): Diffusivity specified from YAML input file
        H (int): Height of the rectangular domain from YAML input file
        flow_rate (int): Input flow rate from YAML file
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    '''
    
    bc_dict.write_bc_dict(geo_directory=case_dir)

    run_flow = RunFlow(
        case_dir=case_dir,
        D=D,
        H=H,
        flow_rate=flow_rate
    )

    # Run CFD
    run_flow.run_cfd()


def run_simulation_from_csv(csv_path: str, mesh_size: float, D: float, vel: float, sources_dir: str, target_size: int=512, row_index: int=0):
    """
    Read parameters from CSV and run simulation for specified row
    
    Parameters:
        csv_path (str): Path to the CSV file with tasks
        mesh_size (str): Mesh size from input YAML file
        D (float): Diffusivity from input YAML file
        vel (float): Velocity from input YAML file
        sources_dir (str): Path to 'sources' directory containing gmshUtilities.py and randSeqAdd.py
        target_size (int): Dimension in pixels of resulting cropped JPEG image
        row_index (int): Which row to process
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
    num_pores = row['a_N']
    radius = row['r_p']
    seed = row['seed']
    
    start = time.time()
    
    # Set inlet flow rate
    H = 1
    flow_rate = vel * H
    
    # Calculate Peclet number
    peclet_number = (radius*vel)/D
    print(f'Peclet number: {peclet_number}')
    
    # Call meshing function
    mesh = MeshGenerator(
        sources_dir=sources_dir,
        max_time_sec=float(60)
    )
    
    # Convert a_N and a_R0 from lists to int and float respectively
    num_pores = int(num_pores[0]) if isinstance(num_pores, list) else int(num_pores)
    radius = float(radius[0]) if isinstance(radius, list) else float(radius)
    
    mesh.generate_mesh(radius=radius, 
         num_pores=num_pores, 
         seed=seed, 
         mesh_size=mesh_size, 
         case_dir=case_dir)
    print(f'Meshing completed at {case_dir}')
    
    # Call CFD function
    run_sim(D=D, 
            H=H, 
            flow_rate=flow_rate, 
            case_dir=case_dir)
    print(f'CFD completed at {case_dir}')
    
    # Convert ParaView vtu to jpeg using Alpine's pvbatch installation
    
    # Flowlab2 pvbatch install path:
    command = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch' # For flowlab2
    # command = 'pvbatch' # For Alpine
    
    # result =subprocess.run([command, 'paraview_convert_alpine.py', case_dir], capture_output=True, text=True) # For Alpine
    
    result = subprocess.run([command, 'paraview_convert_flowlab2.py', case_dir], capture_output=True, text=True) # For Flowlab2

    image_path = os.path.join(case_dir, 'output', os.path.basename(case_dir) + '.jpeg')
    print(f'Image path: {image_path}')
    
    # Crop the jpeg to remove whitespace
    cropped_image_path = crop_image_dynamic.remove_border_whitespace(uncropped_path=image_path)
    
    # Process image and append it to data.csv
    process_images.process_image(image_path=cropped_image_path, case_dir=case_dir)
    
    # Resize images from default to 512x512, store in new folder
    cnn_dir = os.path.join(data_dir, 'cnn_data_512')
    
    # Resize and pad the images in a separate conda environment that has PyTorch installed
    conda_env = 'base'
    command = f'conda run -n {conda_env} python3 pad_images.py {cropped_image_path} {cnn_dir} {case_dir} --target_size {target_size}' 
    subprocess.run(command, shell=True)  
    
    # resize_images(cropped_image_path=cropped_image_path, cnn_dir=cnn_dir, case_dir=case_dir)
    
    # Delete unnecessary files to keep space small
    file_cleanup(case_dir=case_dir)
    
def handle_signal(sig, frame):
    print('Main process interrupted, shutting down...')
    # Ensure all child processes are terminated
    executor.shutdown(wait=False)
    sys.exit(0)
    

# Loop through all rows
if __name__ == "__main__":
    # Set up signal handler to catch Ctrl+C or kill signals
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
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
    
    # Read in sources directory
    sources_dir = config['sources_dir']
    
    # Read in tasks file
    data_dir = config['data_dir']
    tasks_file_name = config['tasks_file_name']
    tasks_file = os.path.join(data_dir, tasks_file_name + '.csv')
    df = pd.read_csv(tasks_file)
    
    num_cores = mp.cpu_count() - 10 # Keep 10 cores free
    
    start_time = datetime.datetime.now()
    
    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to devnull (silences prints)
    sys.stdout = open(os.devnull, 'w')
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks to executor
        futures = {executor.submit(run_simulation_from_csv, tasks_file, mesh_size, D, vel, sources_dir, 512, i) for i in range(len(df))}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Sims'):
            result = future.result()
            results.extens(result)
            
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    days, seconds = total_time.days, total_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print(f'Processed {len(df)} simulations, utilizing {num_cores} cores')
    print(f'Execution time: \n{days} days \n{hours} hours \n{minutes} minutes \n {seconds} seconds')
