import numpy as np
import pandas as pd
import yaml
import itertools
import os
import sys

def write_tasks_file(num_pores_list, radii, seeds, data_dir, file_name, write_csv=False):
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

    if write_csv==True:
        # Save to CSV for use later
        combo_df.to_csv(os.path.join(data_dir, file_name + '.csv'), index=False)
        csv_path = os.path.join(data_dir, file_name + '.csv')
        print(f'CSV file saved to {csv_path}')
    else:
        print('No CSV file written')
     
    return csv_path

def __main__():
    
    try:
        # Configuration YAML file must be set as first argument when calling
        config_file = sys.argv[1]
    except IndexError:
        print("Error: Configuration YAML file must be specified, e.g.: python3 simulate.py /path/to/config_file.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
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
    