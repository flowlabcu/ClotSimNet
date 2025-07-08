import os
import msgpack
import lmdb
import piexif
from PIL import Image
import torch
import io

def create_lmdb(
    data_folder: str, 
    lmdb_path: str
):
    """
    Create an LMDB database from a directory of images with permeability data embedded in each image's EXIF metadata.
    
    Function overview:
    
    Parameters:
        data_folder (str): Path to the directory containing all of the images.
        lmdb_path (str): Path where the LMDB database will be placed.
        
    Returns:
        None
    """
    # Set map size
    env = lmdb.open(lmdb_path, map_size=int(1e10))
    with env.begin(write=True) as txn:
        for idx, filename in enumerate(sorted(os.listdir(data_folder))):
            image_path = os.path.join(data_folder, filename)
            
            # Load image
            image = Image.open(image_path)
            
            # convert image to JPEG bytes
            image_buffer = io.BytesIO()
            image.save(image_buffer, format='JPEG')
            image_bytes = image_buffer.getvalue()
            
            # Load EXIF data
            exif_dict = piexif.load(image.info['exif'])
            
            # Extract and decode permeability value
            if piexif.ExifIFD.UserComment in exif_dict['Exif']:
                metadata_str = exif_dict['Exif'][piexif.ExifIFD.UserComment].decode("utf-8")
                permeability = float(metadata_str)
                # print(permeability)
                
            # Package data into dictionary
            data = {
                'image': image_bytes,
                'permeability': permeability
            }
            
            # Use msgpack to serialize data
            txn.put(f'{idx}'.encode(), msgpack.packb(data, use_bin_type=True))
            
    env.close()
    print(f'LMDB written to {lmdb_path}')
    
# Usage
data_folder = '/mnt/d/clotsimnet_data/clotsimnet_data_test_updated/cnn_data_crop'
lmdb_path = '/home/josh/clotsimnet/data/cnn_data_test_crop.lmdb'

print('Writing LMDB files')
create_lmdb(data_folder=data_folder, lmdb_path=lmdb_path)