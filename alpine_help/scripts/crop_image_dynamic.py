from PIL import Image
import numpy as np
import os

def remove_border_whitespace(uncropped_path: str, threshold: int=245) -> str:
    """
    Automatically crop an image to remove border whitespace while preserving internal content.
    
    Parameters:
        uncropped_path (str): Path to the uncropped image
        threshold (int): RGB value threshold to determine what constitutes "whitespace" (0-255)
        
    Returns:
        output_path (str): Path to the cropped image
    """
    # Open the image
    image = Image.open(uncropped_path)
    
    # Convert to grayscale for easier processing
    gray_image = image.convert('L')
    
    # Convert to numpy array for more precise control
    img_array = np.array(gray_image)
    
    # Find the bounding box of non-white content
    rows = np.any(img_array < threshold, axis=1)
    cols = np.any(img_array < threshold, axis=0)
    
    # Get the boundaries
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # Save the cropped image
    sim_id_image = os.path.splitext(os.path.basename(uncropped_path))[0]

    output_path = os.path.join(os.path.dirname(uncropped_path), f"{sim_id_image}_crop{os.path.splitext(uncropped_path)[1]}")
    cropped_image.save(output_path)
    
    print('Image cropped')

    return output_path
