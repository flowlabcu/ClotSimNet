import sys
import os
import yaml
import pandas as pd
from mesh import MeshGenerator
import time
import argparse

"""
Batch Mesh Generator

Reads a YAML configuration and a task CSV file to generate porous media meshes using `MeshGenerator`. Each case is saved in its specified directory as an `.h5` mesh file.

Usage:
    python3 mesh_all.py path/to/config.yaml
"""

def mesh_all(config_path: str):
    """
    Generates meshes for all cases listed in CSV task file, read from config_path.
    
    Parameters:
        config_path (str): Path to the YAML configuration file. It must contain the following:
            - mesh_size (float): Mesh element size.
            - data_dir (str): Base directory containing tasks CSV file.
            - tasks_file_name (str): CSV file (without extension) containing simulation parameters.
            
    For each row in the tasks CSV file, this function:
        - Extracts the pore radius and the output case directory.
        - Calls `MeshGenerator` to create and save the mesh.
        - Deletes all non-.h5 files in the case directory to conserve disk space.
        
    Returns:
        None
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    mesh_size       = cfg['mesh_size']
    data_dir        = cfg['data_dir']
    tasks_csv       = os.path.join(data_dir, cfg['tasks_file_name'] + '.csv')
    domain          = (1.0, 1.0, 1.0)  # default to unit cube

    df = pd.read_csv(tasks_csv)
    print(f"Generating meshes for {len(df)} cases from {tasks_csv}\n")

    for idx, row in df.iterrows():
        radius   = row['r_p']
        case_dir = row['case_dir']

        # ensure output directory exists
        os.makedirs(case_dir, exist_ok=True)
        
        start = time.time()
        
        print(f"\nProcessing simulation {idx + 1} of {len(df)} \t{round(((idx + 1) / len(df)) * 100, 1)}%")

        # mesh generation
        mesh = MeshGenerator(
            radius    = radius,
            mesh_size = mesh_size,
            domain    = domain,
            case_dir  = case_dir
        )
        h5_path = mesh.write_mesh()
        print(f"[{idx:3d}] Mesh written → {h5_path}")
        
        end = time.time()
        
        print(f'Time taken: {end - start} seconds')
        
        for path, _, files in os.walk(case_dir):
            for name in files:
                if name.endswith('.h5'):
                    print(f'Keeping file: {os.path.join(path, name)}')
                else:
                    # print(f'Would delete file: {os.path.join(path, name)}')
                    os.remove(os.path.join(path, name))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Path to YAML configuration file')
    parser.add_argument(
        'yaml_config_file_path',
        type=str,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    config_file = args.yaml_config_file_path
    
    # Run mesh function
    mesh_all(config_file)
