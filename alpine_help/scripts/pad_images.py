import os
import pandas as pd
from PIL import Image
from torchvision.transforms import v2
import torch.nn.functional as F
import torch
import argparse
import piexif

def resize_images(cropped_image_path: str, cnn_dir: str, case_dir: str, target_size: int=512):
    '''
    Prepare dataset for usage with CNNs. Creates a master data.csv file that links the images and permeability.
    
    NOTE: This file is called with argparse from simulate.py to run in a separate Anaconda environment than the one containing FEniCS due to compatibility issues between the FlowLAB FEniCS environment and the requirements of PyTorch. This separation could be removed if an Anaconda environment is created that has both FEniCS and PyTorch.
    
    Parameters:
        cropped_image_path (str): Path to the cropped image.
        cnn_dir (str): Path to new directory to store all resized images.
        case_dir (str): Path to specific simulation.
        target_size (int): Size to resize images to, in pixels. Resize to square image, resize is target_size x target_size pixels.
    
    Returns:
        None  
    '''

    os.makedirs(cnn_dir, exist_ok=True)
    
    csv_path = os.path.join(case_dir, 'data.csv')
    
    # Open the image and resize it
    image = Image.open(cropped_image_path)
    
    # Calculate the ratio to resize while maintaining aspect ratio
    ratio = image.size[0] / image.size[1]
    
    if ratio > 1:
        new_width = target_size
        new_height = int(target_size / ratio)
        pad_top = (target_size - new_height) // 2
        pad_bottom = target_size - new_height - pad_top
        pad_left = 0
        pad_right = 0
    else:
        new_height = target_size
        new_width = int(target_size * ratio)
        pad_left = (target_size - new_width) // 2
        pad_right = target_size - new_width - pad_left
        pad_top = 0
        pad_right = 0
        
    transform = v2.Compose([
        v2.Resize((new_height, new_width)), # Resize
        v2.Grayscale(num_output_channels=1), # Convert to grayscale
        v2.ToImage(), # Convert to tensor
        v2.ToDtype(torch.float32, scale=True),
    
    ])
    
    image_tensor = transform(image)
    
    padded_tensor = F.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1.0) # 1.0 for white
    
    # Convert to PIL image
    to_pil = v2.ToPILImage()
    padded_image = to_pil(padded_tensor)
    
    # Open CSV and extract permeability
    df = pd.read_csv(csv_path)
    permeability = df['k'].values[0]
    metadata = str(permeability)

    # Create EXIF metadata dictionary to store permeability with image
    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: metadata.encode("utf-8")}}
    exif_bytes = piexif.dump(exif_dict)
    
    # Save the image with metadata
    sim_name = os.path.basename(case_dir)
    save_path = os.path.join(cnn_dir, f'{sim_name}_cnn_{target_size}.jpeg')
    
    # Write padded image with metadata
    padded_image.save(save_path, 'JPEG', quality=100, exif=exif_bytes)

    print(f'Cropped/metadata image saved to {save_path}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images for CNNs.')
    parser.add_argument('cropped_image_path', type=str, help='Path to the cropped image.')
    parser.add_argument('cnn_dir', type=str, help='Path to new directory to store all resized images.')
    parser.add_argument('case_dir', type=str, help='Path to specific simulation.')
    parser.add_argument('--target_size', type=int, default=512, help='Size to resize images to, in pixels. Default is 512.')

    args = parser.parse_args()
    
    resize_images(args.cropped_image_path, args.cnn_dir, args.case_dir, args.target_size)
