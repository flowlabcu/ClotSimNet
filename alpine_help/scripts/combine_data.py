import pandas as pd
import sys
import yaml
import os

def main():
    '''
    Combines all data.csv files across all CFD runs into a single master CSV file for training an MLP.
    
    Parameters:
        config_file (sys.argv[1]): YAML configuration file. Must be specified as command line argument when calling file.
        
    Returns:
        None
    '''
    try:
        # Configuration YAML file must be set as first argument when calling
        config_file = sys.argv[1]
    except IndexError:
        print("Error: Configuration YAML file must be specified, e.g.: python3 combine_data.py /path/to/config_file.yaml")
        sys.exit(1)
        
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
        
    # Pull out highest-level directory to find all data.csv files from all simulations
    data_dir = config['data_dir']
    
    # Get a list of all files with the name 'data.csv' in sub-directories
    filenames = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == 'data.csv':
                filenames.append(os.path.join(root, file))

    # Create an empty list to store your dataframes
    dfs = []

    # Read each file and append its data to the list of dfs
    for filename in filenames:
        df = pd.read_csv(filename)
        dfs.append(df)

    # Concatenate all dfs into one large dataframe
    large_df = pd.concat(dfs, ignore_index=True)
    
    df_cleaned = large_df.dropna(axis=1)

    # Save the large dataframe to a new csv file
    save_path = os.path.join(data_dir, 'mlp_data.csv')
    df_cleaned.to_csv(save_path, index=False)
    print(f'Cleaned MLP data written to {save_path}')
    
if __name__ == '__main__':
    main()
    