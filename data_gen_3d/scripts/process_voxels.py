import skimage as ski
import os
from skimage import io, feature, exposure
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import signal
import pandas as pd
import warnings
import pyvista as pv


def process_vti(
    vti_path: str, 
    case_dir: str
):
    """
    Process a single output VTI voxelated file. Calculates both first-order and second-order/texture features.
    
    Parameters:
        vti_path (str): Path to the VTI file.
        case_dir (str): Path to the specific CFD simulation.
        
    Returns:
        None
    """
    csv_path = os.path.join(case_dir, 'data.csv')
    
    # Create feature dictionary
    features = {}
    
    # Read in VTI file
    grid = pv.read(vti_path)
    
    # Array with data is called "result" when exporting from ParaView
    data = grid.point_data['result']
    
    # Convert the scalar field to a NumPy array
    scalar_array = np.array(grid.point_data['result'])

    # Reshape the scalar array from 1D to 3D. Note that the dimensions from grid match those you set when exporting the data from ParaView as a VTI
    volume = scalar_array.reshape(grid.dimensions, order='F') # Order = F is for Fortran-style order for VTK data
    
    
    # Flatten the voxelated data
    flattened_volume = volume.flatten()
    
    features = {}

    print('Calculating first-order statistics')
    
    # Calculate pixel intensity statistics
    features['img_mean_intensity'] = np.mean(flattened_volume)
    features['img_std_intensity'] = np.std(flattened_volume)
    features['img_min_intensity'] = np.min(flattened_volume)
    features['img_max_intensity'] = np.max(flattened_volume)
    features['img_variance_intensity'] = np.var(flattened_volume)
    features['img_median_intensity'] = np.median(flattened_volume)
    features['img_skewness_intensity'] = skew(flattened_volume)
    features['img_kurtosis_intensity'] = kurtosis(flattened_volume)
    
    print('Finished calculating first-order statistics')
    
    ### Calculate GLCM features ###
    
    # Will compute texture features for each slice, then average each individual feature across all slices
    texture_features_by_slice = {}

    # Distances ranging from both small to large, capturing local and global features
    distances = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

    # Angles of horizontal, vertical, diagonal, anti-diagonal, are standard choices
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    print('Calculating higher-order statistics')
    # Average across the z-axis
    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]
        
        converted_image = ski.util.img_as_ubyte(slice_2d) # Convert to data type for GLCM features
        
        p_graycomatrix = feature.graycomatrix(image=converted_image, distances=distances, angles=angles, symmetric=True, normed=True) # Normed must be done to calculate texture properties
        
        # Use GLCM to find higher-order features
        scalar_properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
        
        for scalar_property in scalar_properties:
            prop_array = feature.graycoprops(p_graycomatrix, scalar_property)
            
            for d_idx, distance in enumerate(distances):
                for a_idx, angle in enumerate(angles):
                    feature_name = f'glcm_{scalar_property}_dist_{distance}_angle_{np.round(angle, 4)}'.replace('.', '_')
                    
                    # Append the current slice feature statistics to the specific list in the texture_features dictionary
                    texture_features_by_slice.setdefault(feature_name, []).append(prop_array[d_idx, a_idx])
                    
                    
                    
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
                texture_features_by_slice.setdefault(f'laws_{name}_mean', []).append(np.mean(np.abs(filtered)))
                texture_features_by_slice.setdefault(f'laws_{name}_std', []).append(np.std(filtered))
                texture_features_by_slice.setdefault(f'laws_{name}_energy', []).append(np.sum(filtered ** 2))
            except RuntimeWarning as e:
                texture_features_by_slice.setdefault(f'laws_{name}_mean', []).append(np.nan)
                texture_features_by_slice.setdefault(f'laws_{name}_std', []).append(np.nan)
                texture_features_by_slice.setdefault(f'laws_{name}_energy', []).append(np.nan)

    print('Finished calculating higher-order statistics')
    
    final_texture_features = {}

    for feature_name, feature_values in texture_features_by_slice.items():
        final_texture_features[feature_name] = np.mean(feature_values)

    # Create one feature dictionary

    features.update(final_texture_features)

    # Convert features to DataFrame and append to CSV
    df = pd.DataFrame([features])
    old_df = pd.read_csv(csv_path)
    output = pd.concat([old_df, df], axis=1)

    # Write to CSV
    output.to_csv(csv_path, index=False)
    print(f'Data written to {csv_path}')
