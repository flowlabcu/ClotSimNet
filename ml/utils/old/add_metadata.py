import os
import pandas as pd
from PIL import Image
from tqdm import tqdm as tqdm
import piexif

def main(cnn_dir: str):
    '''
    Maps permeability to image metadata
    '''
    
    metadata_dir = os.path.join(cnn_dir, 'cnn_512_metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    cnn_data = os.path.join(cnn_dir, 'cnn_data_josh_laptop.csv')
    
    df = pd.read_csv(cnn_data)
    
    images = df.iloc[:, 0]
    permeabilities = df.iloc[:, 1]
    
    for image, permeability in tqdm(zip(images, permeabilities), total=len(images)):
        metadata_str = str(permeability)
        image_name = image
        image = Image.open(image)
        exif_dict = {"Exif": {piexif.ExifIFD.UserComment: metadata_str.encode("utf-8")}}
        
        # Write metadata
        image_id = os.path.basename(image_name)
        exif_bytes = piexif.dump(exif_dict)
        image.save(os.path.join(metadata_dir, image_id), exif=exif_bytes, quality=100)
    print(f'Images with metadata saved to {metadata_dir}')
        
main(cnn_dir='/mnt/d/clotsimnet_data/clotsimnet_data_5k/cnn_data_512')