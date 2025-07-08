import numpy as np
import pandas as pd
import yaml
import itertools
import os
import sys

def write_tasks_file(
    radii: np.ndarray, 
    data_dir: str, file_name: str, write_csv=False
):
    
    print(f'Number of combinations: {len(radii)}')

    combo_df_cols = ['case_dir', 'sim_id', 'r_p']
    
    combo_df = {
    'index': [i for i in range(0, len(radii))],
    'case_dir': [os.path.join(data_dir, f'bcc_lattice_rp_{r:.5f}'.replace('0.', '')) for r in radii],
    'sim_id': [f'bcc_lattice_rp_{r:.5f}'.replace('0.', '') for r in radii],
    'r_p': [r for r in radii],
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
    
    # Pore radii ranges
    pore_radius_low = config['pore_radius_low']
    pore_radius_high = config['pore_radius_high']
    pore_num = config['pore_num']

    
    # File creation and saving
    data_dir = config['data_dir']
    
    file_name = config['tasks_file_name']
    
    write_csv = config['write_csv']
    
    radii = np.linspace(start=pore_radius_low, stop=pore_radius_high, num=pore_num)
    
    # Call function    
    write_tasks_file(radii=radii, data_dir=data_dir, file_name=file_name, write_csv=write_csv)

if __name__ == '__main__':
    __main__()
    