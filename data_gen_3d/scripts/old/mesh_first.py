import pandas as pd
import time
import os
import shutil
import subprocess
import yaml
import sys
from mesh import MeshGenerator

def mesh_single(csv_path: str, mesh_size: float, sources_dir: str, row_index: int=0):
    """
    Read parameters from tasks CSV file and run meshing for specified row
    
    Parameters:
        csv_path (str): Path to the CSV file with tasks
        mesh_size (str): Mesh size from input YAML file
        sources_dir (str): Path to 'sources' directory containing gmshUtilities.py and randSeqAdd.py
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
    
    # Call meshing function
    mesh = MeshGenerator(
        sources_dir=sources_dir,
        max_time_sec=float(300)
    )
    
    # Convert a_N and a_R0 from lists to int and float respectively
    num_pores = int(num_pores[0]) if isinstance(num_pores, list) else int(num_pores)
    radius = float(radius[0]) if isinstance(radius, list) else float(radius)
    
    print('Meshing started')
    mesh.generate_mesh(radius=radius, 
         num_pores=num_pores, 
         seed=seed, 
         mesh_size=mesh_size, 
         case_dir=case_dir)
    print(f'Meshing completed at {case_dir}')
    
    
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
    
    ### ---- Pull out CFD parameters from YAML configuration file ---- ###
    
    mesh_size = config['mesh_size']
    
    # Read in sources directory
    sources_dir = config['sources_dir']
    
    # Read in tasks file
    data_dir = config['data_dir']
    tasks_file_name = config['tasks_file_name']
    tasks_file = os.path.join(data_dir, tasks_file_name + '.csv')
    df = pd.read_csv(tasks_file)
    
    for i in range(len(df)):
        print(f"\nMeshing case {i + 1} of {len(df)} \t{round(((i + 1) / len(df)) * 100, 1)}%")
        
        start = time.time()
        
        try:
            mesh_single(csv_path=tasks_file, mesh_size=mesh_size, sources_dir=sources_dir, row_index=i)
        
        except Exception as e:
            print(f'Error meshing case {i+1}: {str(e)}')
        
        end = time.time()
        
        print(f'Meshing finished in {end-start} seconds')
