import pandas as pd
import time
import os
import argparse
import shutil
from flow_steady import RunFlow
import geo_to_bc_dict as bc_dict
import subprocess
import yaml
import sys
import multiprocessing
import crop_image_dynamic
import process_images
from mesh import MeshGenerator

def file_cleanup(
    case_dir: str
):
    """
    Deletes unneeded files to keep dataset size as small as possible.
    
    Parameters:
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    """
    for root, _, files in os.walk(case_dir):
        for file in files:
            if not file.endswith(('.csv', '.jpeg')) and 'cnn' not in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            else:
                file_path = os.path.join(root, file)
    
    # Remove empty "boundaries" directory
    boundaries_dir = os.path.join(case_dir, 'boundaries')
    if os.path.isdir(boundaries_dir) and not os.listdir(boundaries_dir):
        os.rmdir(boundaries_dir)
        
    # Remove non-empty 'output' directory
    output_dir = os.path.join(case_dir, 'output')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    print("File cleanup complete.")
    
def run_sim(
    D: float, 
    H: int, 
    flow_rate: int, 
    case_dir: str
):
    """
    Run the CFD simulation
    
    Parameters:
        D (float): Diffusivity specified from YAML input file
        H (int): Height of the rectangular domain from YAML input file
        flow_rate (int): Input flow rate from YAML file
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    """
    bc_dict.write_bc_dict(geo_directory=case_dir)

    run_flow = RunFlow(
        case_dir=case_dir,
        D=D,
        H=H,
        flow_rate=flow_rate
    )
    # Run CFD
    run_flow.run_cfd()

def run_simulation_from_csv(
    csv_path: str, 
    mesh_size: float, 
    D: float, 
    vel: float,
    sources_dir: str, 
    data_dir: str, 
    target_size: int=512,
    row_index: int=0
):
    """
    Runs a single CFD simulation based on parameters from a row in the CSV task file.
    
    Parameters:
        csv_path (str): Path to the CSV file with simulation tasks.
        mesh_size (float): Mesh element size from input YAML file for mesh generation.
        D (float): Diffusivity from input YAML file for each CFD simulation.
        vel (float): Peak inlet velocity from input YAML file for each CFD simulation.
        sources_dir (str): Path to 'sources' directory containing gmshUtilities.py and randSeqAdd.py.
        data_dir (str): Base directory where data is stored (used to construct output paths).
        target_size (int): Dimension in pixels of resulting cropped JPEG image. Defaults to 512x512
        row_index (int): Index of the row in the CSV simulation task file to process. Defaults to 0 (the beginning of the file)
        
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
    peclet_number = (radius * vel) / D
    print(f'Peclet number: {peclet_number}')
    
    # Call meshing function
    mesh = MeshGenerator(
        sources_dir=sources_dir,
        max_time_sec=float(60)
    )
    
    # Convert a_N and r_p from lists to int and float if needed
    num_pores = int(num_pores[0]) if isinstance(num_pores, list) else int(num_pores)
    radius = float(radius[0]) if isinstance(radius, list) else float(radius)
    
    mesh.generate_mesh(radius=radius, 
                         num_pores=num_pores, 
                         seed=seed, 
                         mesh_size=mesh_size, 
                         case_dir=case_dir)
    print(f'Meshing completed at {case_dir}')
    
    # Call CFD function
    run_sim(D=D, H=H, flow_rate=flow_rate, case_dir=case_dir)
    print(f'CFD completed at {case_dir}')
    
    # Convert ParaView vtu to jpeg using your pvbatch installation
    command = '/home/joshgregory/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch'  # For flowlab2
    # command = 'pvbatch'  # For Alpine
    result = subprocess.run([command, 'paraview_convert_flowlab2.py', case_dir],
                            capture_output=True, text=True)
    
    image_path = os.path.join(case_dir, 'output', os.path.basename(case_dir) + '.jpeg')
    print(f'Image path: {image_path}')
    
    # Crop the jpeg to remove whitespace
    cropped_image_path = crop_image_dynamic.remove_border_whitespace(uncropped_path=image_path)
    
    # Process image and append data to CSV
    process_images.process_image(image_path=cropped_image_path, case_dir=case_dir)
    
    # Resize images from default to 512x512, store in new folders using data_dir
    cnn_dir = os.path.join(data_dir, 'cnn_data_center_crop')
    crop_dir = os.path.join(data_dir, 'cnn_data_crop')
    
    # Resize and pad images in a separate conda environment that has PyTorch installed
    conda_env = 'base'
    command = f'conda run -n {conda_env} python3 pad_images.py {cropped_image_path} {cnn_dir} {case_dir} --target_size {target_size}' 
    subprocess.run(command, shell=True)
    
    command = f'conda run -n {conda_env} python3 no_crop.py {cropped_image_path} {crop_dir} {case_dir}' 
    subprocess.run(command, shell=True)   
    
    # Delete unnecessary files to keep space small
    file_cleanup(case_dir=case_dir)
    
def simulation_task_wrapper(
    row_index: int, 
    csv_path: str, 
    mesh_size: float, 
    D: float, 
    vel: float, 
    sources_dir: str, 
    data_dir: str, 
    target_size: int
):
    """
    Safely runs a single simulation task with error handling.
    
    Wraps `run_simulation_from_csv()` and prints any errors that occur during execution.
    
    Parameters:
        row_index (int): Index of the row in the CSV simulation task file to process.
        csv_path (str): Path to the CSV simulation task file.
        mesh_size (float): Mesh element size.
        D (float): Diffusivity value.
        vel (float): Inlet velocity.
        sources_dir (str): Path to 'sources' directory containing gmshUtilities.py and randSeqAdd.py.
        data_dir (str): Directory for storing outputs.
        target_size (int): Target image size in pixels.
        
    Return:
        None
    """
    try:
        run_simulation_from_csv(csv_path, mesh_size, D, vel, sources_dir, data_dir, target_size, row_index)
    except Exception as e:
        print(f"Simulation for row {row_index} encountered an error: {e}")

def main():
    """
    Runs a batch of CFD simulations and processes the output images.
    
    Calls simulation_task_wrapper, which in turn calls run_simulation_from_csv, but with a time limit. 
    """
    # Add argument parser for path to YAML configuration file
    parser = argparse.ArgumentParser(description='Path to YAML configuration file (same one as used in write_tasks.py)')
    parser.add_argument(
        'yaml_config_file_path',
        type=str,
        help='Path to YAML configuration file (same one as used in write_tasks.py)'
    )
    
    args = parser.parse_args()
    
    config_file = args.yaml_config_file_path
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    ### ---- Pull out CFD parameters from YAML configuration file ---- ###
    mesh_size = config['mesh_size']
    D = config['diffusivity']
    vel = config['inlet_velocity']
    
    # Read in sources directory from YAML.
    sources_dir = config['sources_dir']
    
    # Read in data directory and tasks file name from YAML.
    data_dir = config['data_dir']
    tasks_file_name = config['tasks_file_name']
    tasks_file = os.path.join(data_dir, tasks_file_name + '.csv')
    df = pd.read_csv(tasks_file)
    target_size = 512
    timeout_seconds = 120
    
    for row_index in range(len(df)):
        print(f"\nProcessing simulation {row_index + 1} of {len(df)} \t{round(((row_index + 1) / len(df)) * 100, 1)}%")
        
        p = multiprocessing.Process(
            target=simulation_task_wrapper,
            args=(row_index, tasks_file, mesh_size, D, vel, sources_dir, data_dir, target_size)
        )
        
        p.start()
        p.join(timeout=timeout_seconds)
        
        if p.is_alive():
            print(f"Task {row_index} exceeded {timeout_seconds} seconds. Terminating process...")
            p.terminate()
            p.join()  # Ensure the process is cleaned up
            
            # Kill lingering gmsh processes
            print("Killing lingering gmsh processes...")
            subprocess.run("pkill gmsh", shell=True)
            
        print(f"Finished processing task {row_index}. Moving on to next simulation.")
        
if __name__ == '__main__':
    main()