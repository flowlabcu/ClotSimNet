import os
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
import wandb
import lmdb
import msgpack
import sys
import torchvision
import onnxruntime as ort
# torch.set_float32_matmul_precision('highest') # Options are medium, high, highest


model_path = '/home/josh/ClotSimNet-Models/base/mlp_o1/mlp_o1_base.onnx'
data_path = '/home/josh/clotsimnet/data/mlp_data_test.csv'

# Load ONNX model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

data = pd.read_csv(data_path)

data = data.dropna(axis=1) # Remove NaN values
# Get feature columns
start_idx = data.columns.get_loc('img_mean_intensity')
end_idx = data.columns.get_loc('img_kurtosis_intensity')

feature_cols = data.columns[start_idx:end_idx+1]

input_data = data[feature_cols].values.astype('float32')

labels = data['k'].values.astype('float32').flatten()

scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Pull file name
if 'id' in data.columns:
    filenames = data['id'].values

results = []
for i in range(len(input_data)):
    sample = input_data[i:i+1]
    permeability = labels[i]
    output = session.run(None, {input_name: sample})[0]
    pred = float(np.exp(output)[0])
    results.append({
        'filename': os.path.basename(filenames[i]),
        'permeability': permeability,
        'prediction': pred
    })

results_df = pd.DataFrame(results)

pd.options.display.float_format = '{:.6e}'.format
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('inference_mlp_o1_base_onnx.csv', index=False, float_format='%.6e')
