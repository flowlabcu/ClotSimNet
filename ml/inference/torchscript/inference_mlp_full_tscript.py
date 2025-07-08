import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torch
# torch.set_float32_matmul_precision('highest') # Options are medium, high, highest


model_path = '/home/josh/clotsimnet/ml/models/mlp_base/mlp_full_base_tscript.pt'
data_path = '/home/josh/clotsimnet/data/test_dataset/mlp_data_test_dataset.csv'

# Load TorchScript model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path, map_location=device)
model.eval() # Set model to evaluation mode

data = pd.read_csv(data_path)

data = data.dropna(axis=1) # Remove NaN values
# Get feature columns
start_idx = data.columns.get_loc('img_mean_intensity')
feature_cols = data.columns[start_idx:]

input_data = data[feature_cols].values.astype('float32')
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

labels = data['k'].values.astype('float32').reshape(-1,1)

# Pull file name
if 'id' in input_data.columns:
    filenames = data[id].values

# Process one sample at a time
results = []
with torch.no_grad(): # Disable gradient calculation for inference
    for i in range(len(input_data)):
        sample = torch.tensor(input_data[i:i+1], dtype=torch.float32, device=device)
        
        permeability = labels[i]
        
        output = model(sample).cpu().numpy()[0][0]
        pred = np.exp(output)
        results.append({
            'filename': filenames[i],
            'permeability': permeability,
            'prediction': pred
        })

results_df = pd.DataFrame(results)
print(results_df)
