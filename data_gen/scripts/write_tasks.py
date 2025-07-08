import numpy as np
import pandas as pd
import yaml
import itertools
import os
import sys
import argparse

def write_tasks_file(
    num_pores_list: np.ndarray, 
    radii: np.ndarray, 
    seeds: np.ndarray, 
    data_dir: str, 
    file_name: str, 
    write_csv: bool=False
):
    """
    Generates a parametric combination of all pore counds, radii, and seeds. Saves them as a CSV file in data_dir in a file named file_name
    
    Parameters:
        num_pores_list (np.ndarray): Array of pore count values from YAML configuration file ("Number of pores").
        radii (np.ndarray): Array of pore radius values from YAML configuration file ("Pore radii").
        seeds (np.ndarray): Array of random seed values. Seed value 2 is automatically excluded.
        data_dir (str): Directory path where the output CSV will be saved.
        file_name (str): Name of the output CSV file without extension.
        write_csv (bool, optional): If True, writes the combinations to a CSV file. Defaults to False.
        
    Returns:
        csv_path (str): Full path to the written CSV file if `write_csv` is True; otherwise returns None.
        
    Notes:
        - Seed value of 2 is automatically excluded due to known issues with Gmsh.
        - The directory `data_dir` is automatically created if it does not exist.
        - The CSV includes columns named `index`, `case_dir`, `sim_id`, `a_N`, `r_p`, `seed`.
    """
    # Exclude the number 2 because that breaks GMesh for some reason
    seeds = seeds[seeds != 2]
    
    combinations = list(itertools.product(num_pores_list, radii, seeds))

    print(f'Number of combinations: {len(combinations)}')

    combo_df_cols = ['case_dir', 'a_N', 'r_p', 'seed']
    
    combo_df = {
    'index': [i for i in range(0, len(combinations))],
    'case_dir': [os.path.join(data_dir, f'aN_{c[0]}_rp_{c[1]:.5f}'.replace('0.', '') + f'_seed_{c[2]}') for c in combinations],
    'sim_id': [f'aN_{c[0]}_rp_{c[1]:.5f}'.replace('0.', '') + f'_seed_{c[2]}' for c in combinations],
    'a_N': [c[0] for c in combinations],
    'r_p': [c[1] for c in combinations],
    'seed': [c[2] for c in combinations],
}

    combo_df = pd.DataFrame(combo_df)

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Save to CSV for use later
    combo_df.to_csv(os.path.join(data_dir, file_name + '.csv'), index=False)
    csv_path = os.path.join(data_dir, file_name + '.csv')
    print(f'CSV file saved to {csv_path}')
     
    return csv_path

def __main__():
    """
    Parses the path to YAML configuration file, loads simulation parameters from the file, constructs arrays of pore numbers, pore radii, and seeds, and calls the `write_tasks_file` function to generate combinations and optionally write them to disk.
    
    Command-line Arguments:
        yaml_config_file_path (str): Path to the YAML configuration file.
        
    Effects:
        - Loads and parses YAML config. gfile
        - Prepares pore counts, radii, and seeds as NumPy arrays.
        - Calls `write_tasks_file()` with the prepared inputs.
    """
    # Add argument parser for path to YAML configuration file
    parser = argparse.ArgumentParser(description='Path to YAML configuration file')
    parser.add_argument(
        'yaml_config_file_path',
        type=str,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    config_file = args.yaml_config_file_path
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    ## Assign values from the configuration YAML file ##
    
    # Number of pore ranges
    pore_num_low = config['pore_num_low']
    pore_num_high = config['pore_num_high']
    
    # Pore radii ranges
    pore_radius_low = config['pore_radius_low']
    pore_radius_high = config['pore_radius_high']
    pore_num = config['pore_num']
    
    # Seed ranges
    seed_low = config['seed_low']
    seed_high = config['seed_high']
    
    # File creation and saving
    data_dir = config['data_dir']
    
    file_name = config['tasks_file_name']
    
    write_csv = config['write_csv']
    
    # Create everything to pass to function
    num_pores_list = np.arange(pore_num_low, pore_num_high)
    
    radii = np.linspace(start=pore_radius_low, stop=pore_radius_high, num=pore_num)
    
    seeds = np.arange(start=seed_low, stop=seed_high)
    
    # Call function    
    write_tasks_file(num_pores_list=num_pores_list, radii=radii, seeds=seeds, data_dir=data_dir, file_name=file_name, write_csv=write_csv)

if __name__ == '__main__':
    __main__()
    