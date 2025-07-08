import pandas as pd
import sys
import yaml
import os
import argparse

def main():
    """
    Combines all data.csv files across all CFD runs into a single master CSV file for training an MLP.
    
    Parameters:
        data_dir (command-line argument): Path to data directory containing all simulations results.
        
    Returns:
        None
    """
   # Add argument parser for path to YAML configuration file
    parser = argparse.ArgumentParser(description='Path to data directory with simulation results')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to data directory with simulation results'
    )
        
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
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
