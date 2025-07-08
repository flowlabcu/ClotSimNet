import os
import pandas as pd
from PIL import Image, PngImagePlugin
import tqdm

def create_dataset(cnn_dir: str, cnn_csv: str='cnn_data.csv', extension: str='.jpeg'):
    """
    Takes in the entire CNN dataset (images only), creates a CSV dataset with paths to images and their associated permeabilities
    
    Parameters:
        cnn_dir (str): Path to the directory where the images for the CNN are stored.
        cnn_csv (str): Name for the CNN CSV file.
        extension (str): File extension of the CNN images.
        
    Returns:
        None
    """
    for file in tqdm.tqdm(len(os.listdir(cnn_dir))):
        data = []
        if file.endswith(extension):
            file_path = os.path.join(cnn_dir, file)
            with Image.open(file_path) as image:
                permeability = image.info
        
        data.append([file_path, permeability])
    
    # Save as a single CSV file
    df = pd.DataFrame(data, columns=['image_path', 'permeability'])
    save_path = os.path.join(cnn_dir, cnn_csv)
    # os.path.join(root_dir, output_csv)
    df.to_csv(save_path, index=False)
    print(f'CNN data CSV saved to {save_path}')
