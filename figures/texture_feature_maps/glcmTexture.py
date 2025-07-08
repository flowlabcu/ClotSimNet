import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows

"""
glcm_feature_extraction.py

This script performs Gray-Level Co-occurrence Matrix (GLCM) based texture feature extraction 
on grayscale images, and visualizes selected Haralick features such as homogeneity and energy.
Saves the resulting visualizations to `glcm_map.svg`

Usage:
    python3 glcm_feature_extraction.py

Notes:
- Works best on grayscale or single-channel float images.
- Feature values can be sensitive to image dynamic range; normalization may help.
- Padding strategies (`reflect`, `edge`) affect windowed feature outputs.

Authors:
Debanjan Mukherjee, adapted by Josh Gregory
"""

def injectRandomNoise(image: np.ndarray):
    """
    Adds small amounts of Gaussian noise to an image.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        
    Returns:
        np.ndarray: Output image with added Gaussian noise.
    """
    image += np.random.normal(0, 1e-6, image.shape)
    return image

# --- Compute GLCM and Haralick Features ---
def compute_glcm_features_no_padding(image, distances=[1], angles=[0]):
    """
    Computes GLCM features without padding
    
    Parameters:
        image (np.ndarray): Grayscale input image. Convered to uint8 if needed.
        distances (list of int): List of pixel distances for GLCM computation.
        angles (list of float): List of angles (in radians) for GLCM computation
        
    Returns:
        tuple of NumPy arrays containing calculated values for contrast, dissimilarity, homogeneity, energy, and correlation from GLCM.
    """
    # Ensure the image is uint8 for GLCM calculation
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image

    # GLCM calculation
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract Haralick features
    contrast = graycoprops(glcm, 'contrast').squeeze()
    dissimilarity = graycoprops(glcm, 'dissimilarity').squeeze()
    homogeneity = graycoprops(glcm, 'homogeneity').squeeze()
    energy = graycoprops(glcm, 'energy').squeeze()
    correlation = graycoprops(glcm, 'correlation').squeeze()

    return contrast, dissimilarity, homogeneity, energy, correlation

def compute_glcm_features(image, distances=[1], angles=[0]):
    """
    Computes GLCM features with padding and crops them back to their original size.
    
    Parameters:
        image (np.ndarray): Grayscale input image. Convered to uint8 if needed.
        distances (list of int): List of pixel distances for GLCM computation.
        angles (list of float): List of angles (in radians) for GLCM computation
        
    Returns:
        tuple of NumPy arrays containing calculated values for contrast, dissimilarity, homogeneity, energy, and correlation from GLCM.
    """
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image

    # Pad image to ensure output matches original size
    pad_width = max(distances)
    padded_image = np.pad(image, pad_width, mode='edge')

    # GLCM calculation
    glcm = graycomatrix(padded_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract Haralick features
    contrast = graycoprops(glcm, 'contrast').squeeze()
    dissimilarity = graycoprops(glcm, 'dissimilarity').squeeze()
    homogeneity = graycoprops(glcm, 'homogeneity').squeeze()
    energy = graycoprops(glcm, 'energy').squeeze()
    correlation = graycoprops(glcm, 'correlation').squeeze()

    # Crop features back to original image size
    contrast = contrast[pad_width:-pad_width, pad_width:-pad_width]
    dissimilarity = dissimilarity[pad_width:-pad_width, pad_width:-pad_width]
    homogeneity = homogeneity[pad_width:-pad_width, pad_width:-pad_width]
    energy = energy[pad_width:-pad_width, pad_width:-pad_width]
    correlation = correlation[pad_width:-pad_width, pad_width:-pad_width]

    return contrast, dissimilarity, homogeneity, energy, correlation

def compute_glcm_features_window(image, distances=[1], angles=[0], window_size=5):
    """
    Computes GLCM features using a sliding window.
    
    Parameters:
        image (np.ndarray): Grayscale input image. Convered to uint8 if needed.
        distances (list of int): List of pixel distances for GLCM computation.
        angles (list of float): List of angles (in radians) for GLCM computation.
        window_size (int): Size of the local window.
        
    Returns:
        tuple of NumPy arrays containing calculated values for contrast, dissimilarity, homogeneity, energy, and correlation from GLCM.
    """
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image

    # Pad image to avoid boundary issues
    pad_width = window_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')

    # Initialize feature maps
    contrast = np.zeros_like(image, dtype=float)
    dissimilarity = np.zeros_like(image, dtype=float)
    homogeneity = np.zeros_like(image, dtype=float)
    energy = np.zeros_like(image, dtype=float)
    correlation = np.zeros_like(image, dtype=float)

    # Sliding window approach
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]

            # Compute GLCM for the local window
            glcm = graycomatrix(window, distances=distances, angles=angles, symmetric=True, normed=True)

            # Extract features
            contrast[i, j] = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity[i, j] = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]
            energy[i, j] = graycoprops(glcm, 'energy')[0, 0]
            correlation[i, j] = graycoprops(glcm, 'correlation')[0, 0]

    return contrast, dissimilarity, homogeneity, energy, correlation

def compute_glcm_features_eff_window(image, distances=[1], angles=[0], window_size=5):
    """
    Computes GLCM features using a rolling window.
    
    Parameters:
        image (np.ndarray): Grayscale input image. Convered to uint8 if needed.
        distances (list of int): List of pixel distances for GLCM computation.
        angles (list of float): List of angles (in radians) for GLCM computation.
        window_size (int): Size of the local window.
        
    Returns:
        tuple of NumPy arrays containing calculated values for homogeneity and energy from GLCM.
    """
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image
    
    # Pad image to avoid boundary issues
    pad_width = window_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')

    # Create rolling window view
    windows = view_as_windows(padded_image, (window_size, window_size))

    # Initialize feature maps
    # contrast = np.zeros(image.shape, dtype=float)
    # dissimilarity = np.zeros(image.shape, dtype=float)
    # entropy = np.zeros(image.shape, dtype=float)
    homogeneity = np.zeros(image.shape, dtype=float)
    # mean = np.zeros(image.shape, dtype=float)
    energy = np.zeros(image.shape, dtype=float)
    # correlation = np.zeros(image.shape, dtype=float)

    # Vectorized computation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = windows[i, j]

            # Compute GLCM for the window
            glcm = graycomatrix(window, distances=distances, angles=angles, symmetric=True, normed=True)

            # Extract features
            # contrast[i, j] = graycoprops(glcm, 'contrast')[0, 0]
            # dissimilarity[i, j] = graycoprops(glcm, 'dissimilarity')[0, 0]
            # entropy[i, j] = graycoprops(glcm, 'entropy')[0, 0]
            homogeneity[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]
            # mean[i, j] = graycoprops(glcm, 'mean')[0, 0]
            energy[i, j] = graycoprops(glcm, 'energy')[0, 0]
            # correlation[i, j] = graycoprops(glcm, 'correlation')[0, 0]

    # return contrast, dissimilarity, homogeneity, energy, correlation
    return homogeneity, energy

def visualize_glcm_features(image, homogeneity, energy):
    """
    Displays the input image as well as homogeneity and energy maps.
    
    Saves the figure to `glcm_map.svg`.
    
    Parameters:
        image (np.ndarray): NumPy array of original image
        homogeneity (np.ndarray): GLCM homogeneity map.
        energy (np.ndarray): GLCM energy map.
    
    Returns:
        None
    """
    
    # 1) recreate fig & axes
    fig, axes = plt.subplots(1, 3, figsize=(25, 20))
    features = [image, homogeneity, energy]
    titles   = ['Original Image', 'Homogeneity', 'Energy']
    cmaps    = ['gray', 'viridis', 'plasma']

    for ax, data, title, cmap in zip(axes, features, titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.axis('off')

        # 2) grab position of this image-axis
        pos = ax.get_position()
        cbar_height = 0.03
        pad = 0.02
        cbar_bottom = pos.y0 - pad - cbar_height

        # 3) add a new axis below it
        cax = fig.add_axes([pos.x0, cbar_bottom, pos.width, cbar_height])

        # 4) horizontal colorbar with bold 24pt ticks
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=24)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontweight('bold')
            
    plt.savefig('glcm_map.svg', bbox_inches='tight')
    


# --- Example Usage ---
image = io.imread('/home/josh/clotsimnet/data/cnn_data_test/cnn_data_crop/aN_447_rp_01700_seed_500_crop.jpeg').squeeze()#, as_gray=True)
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)

homogeneity, energy = compute_glcm_features_eff_window(image)
visualize_glcm_features(image, homogeneity, energy)

