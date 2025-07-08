'''
File to convert paths from one machine to another.

Example is if data is generated on Flowlab2, but ML is run on Alpine. Paths to files are different between these machines. This file translates them from one to another to allow for the ML algorithms to associate permeabilities with images correctly. Otherwise, permeabilities and images couldn't be linked due to image file paths not being correct.
'''
import os
import pandas as pd

def convert_paths(csv_path: str, current_path: str, desired_path: str, output_suffix: str):
    '''
    Converts paths in CNN dataset CSV file from one machine to another.
    
    DO NOT specify '.csv' when calling this method, it is automatically added to the new file.
    
    Parameters:
        csv_path (str): Path to CNN CSV containing image paths and permeabilities
        current_path (str): Path prefix in the current file
        desired_path (str): Desired path prefix
        output_suffix (str): New name for CNN CSV file
    '''
    if '.csv' in csv_path:
        raise ValueError("ERROR: The CSV file path should not contain '.csv'. Please remove it from the path.")
    
    df = pd.read_csv(f'{csv_path}.csv')
    
    # Ensure the first column is the file path column and update it
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x.replace(current_path, desired_path) if isinstance(x, str) else x)
    
    # Save the modified DataFrame back to a new CSV
    save_path = f"{csv_path}_{output_suffix}.csv"
    df.to_csv(save_path, index=False)
    print(f"File paths updated and saved to {save_path}")
    

convert_paths(csv_path='/mnt/d/clotsimnet_data/clotsimnet_data_5k/cnn_data_512/cnn_data', current_path='/mnt/hdd1/joshgregory', desired_path='/mnt/d/clotsimnet_data', output_suffix='josh_laptop')
