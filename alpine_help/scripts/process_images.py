# NOTE: IMAGE EXTRACTION SHOULD HAVE IMAGES WITH INTENSITIES FROM 0-255. IF USING IMAGES AS RAW INPUT WANT SCALING 0-1.

import skimage as ski
# import h5py
import multiprocessing as mp
import os
import glob
from skimage import io, img_as_float, feature, exposure
from skimage.color import rgb2gray
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import signal
from concurrent.futures import as_completed
import concurrent
from tqdm import tqdm
import pandas as pd
import warnings

def process_image(image_path, case_dir):
    
    csv_path = os.path.join(case_dir, 'data.csv')
    
    # Create feature dictionary
    features = {}
    # features['image_path'] = image_path
    
    # Read in image
    image = io.imread(image_path)
    
    # Convert image to grayscale
    gray_image = rgb2gray(image)
    
    converted_image = ski.util.img_as_ubyte(gray_image) # Set to proper dtype for 0-255, see https://scikit-image.org/docs/stable/user_guide/data_types.html
    
    # Flatten image
    flattened_image = converted_image.flatten()
    
    # Calculate pixel intensity statistics
    features['img_mean_intensity'] = np.mean(flattened_image)
    features['img_std_intensity'] = np.std(flattened_image)
    features['img_min_intensity'] = np.min(flattened_image)
    features['img_max_intensity'] = np.max(flattened_image)
    features['img_variance_intensity'] = np.var(flattened_image)
    features['img_median_intensity'] = np.median(flattened_image)
    features['img_skewness_intensity'] = skew(flattened_image)
    features['img_kurtosis_intensity'] = kurtosis(flattened_image)
    
    ### Calculate GLCM features ###
    
    # Distances ranging from both small to large, capturing local and global features
    distances = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    
    # Angles of horizontal, vertical, diagonal, anti-diagonal, are standard choices
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Calculate GLCM
    p_graycomatrix = feature.graycomatrix(image=converted_image, distances=distances, angles=angles, symmetric=True, normed=True) # Normed must be done to calculate texture properties
    
    # Use GLCM to find higher-order features
    scalar_properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    
    for scalar_property in scalar_properties:
        
        prop_array = feature.graycoprops(p_graycomatrix, scalar_property)
        
        for d_idx, distance in enumerate(distances):
            
            for a_idx, angle in enumerate(angles):
                
                features[f'glcm_{scalar_property}_dist_{distance}_angle_{np.round(angle, 4)}'.replace('.', '_')] = prop_array[d_idx, a_idx]
        
    # Laws convolutional kernels, taken from https://www.sciencedirect.com/science/article/pii/B9780124095458000029
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([1, 0, 2, 0, -1])
    W5 = np.array([-1, 2, 0, -2, 1])
    R5 = np.array([1, -4, 6, -4, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        kernels = {
            'L5E5_div_E5L5': np.divide( np.outer(L5, E5), np.outer(E5, L5)),
            'L5R5_div_R5L5': np.divide( np.outer(L5, R5), np.outer(R5, L5)),
            'E5S5_div_S5E5': np.divide( np.outer(E5, S5), np.outer(S5, E5)),
            'S5S5': np.outer(S5, S5),
            'R5R5': np.outer(R5, R5),
            'L5S5_div_S5L5': np.divide( np.outer(L5, S5),  np.outer(S5, L5)),
            'E5E5': np.outer(E5, E5),
            'E5R5_div_R5E5': np.divide( np.outer(E5, R5), np.outer(R5, E5) ),
            'S5R5_div_R5S5': np.divide( np.outer(S5, R5), np.outer(R5, S5))
        }
    
    # Suppress divide by zero warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
    
    for name, kernel in kernels.items():
        try:
            filtered = signal.convolve2d(converted_image, kernel)
            features[f'laws_{name}_mean'] = np.mean(np.abs(filtered))
            features[f'laws_{name}_std'] = np.std(filtered)
            features[f'laws_{name}_energy'] = np.sum(filtered ** 2)
        except RuntimeWarning as e:
            features[f'laws_{name}_mean'] = np.nan
            features[f'laws_{name}_std'] = np.nan
            features[f'laws_{name}_energy'] = np.nan
        
    # Convert features to DataFrame and append to CSV
    df = pd.DataFrame([features])
    old_df = pd.read_csv(csv_path)
    output = pd.concat([old_df, df], axis=1)
    
    # Write to CSV
    output.to_csv(csv_path, index=False)
    print(f'Data written to {csv_path}')

# sim_id = 'aN_454_rp_024_seed_1'

# image_path_use = os.path.join('/home/josh/test_data_16', f'{sim_id}', 'output', f'{sim_id}_crop.jpeg')
# output_dir_use = os.path.join('/home/josh/test_data_16', f'{sim_id}')

# process_image(image_path=image_path_use, output_dir=output_dir_use)
