import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def make_dendrogram(csv_path: str):
    """
    Create dendrogram from all image features. Saves resulting dendrogram in SVG format.
    
    Parameters:
        csv_path (str): Path to CSV dataset with extracted image features.
        
    Returns:
        None
    """
    df = pd.read_csv(csv_path)

    # Replace NaN and infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    features = df.iloc[:, 5:]  # Skip image names and CFD data
    feature_names = features.columns.tolist()
    # print(feature_names)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute distance matrix and linkage
    distance_matrix = pdist(features_scaled.T, metric='euclidean')  
    linkage_matrix = sch.linkage(distance_matrix, method='ward')

    # Create figure
    plt.figure(figsize=(8, 6))

    # Generate dendrogram and capture the leaf positions
    dendrogram = sch.dendrogram(linkage_matrix, leaf_rotation=90, no_labels=True)

    # Define custom group positions
    group_positions = [450, 1550, 2375]  # Adjust these based on dendrogram layout
    group_labels = ['Group A', 'Group B', 'Group C']

    # Manually place text labels at x positions
    for pos, label in zip(group_positions, group_labels):
        plt.text(pos, -50, label, fontsize=12, fontweight='bold', ha='center', transform=plt.gca().transData)

    plt.ylabel('Euclidean Distance', weight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)
    plt.title('Hierarchical Clustering of Image Features', weight='bold', fontsize=14)

    # Adjust layout so labels are visible
    plt.tight_layout()
    # plt.show()
    plt.savefig('dendrogram.svg') # Can change the file name if desired
    
    ddata = sch.dendrogram(linkage_matrix, labels=feature_names, no_plot=True)
    
    leaves = list(zip(ddata['ivl'], ddata['leaves_color_list']))
    
    by_color = defaultdict(list)
    for label, color in leaves:
        by_color[color].append(label)
        
    # for color, feats in by_color.items():
    #     print(f"{color}: {len(feats)} features → {feats[:5]}")
        
    # print(by_color['C3'])
    

make_dendrogram(csv_path='/home/josh/clotsimnet/data/mlp_data_5k.csv')
