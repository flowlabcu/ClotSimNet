import os
import sys
import torch
import pandas as pd
import numpy as np
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Adjust these paths as needed ──
STATE_PATH = '/home/josh/ClotSimNet-Models/base/mlp_full/mlp_full_base.safetensors'
DATA_PATH  = '/home/josh/clotsimnet/data/mlp_data_aug_cnn_pipe.csv'
save_file = 'mlp_full_base_cnn_pipe_preds.csv'
# '/home/josh/clotsimnet/data/mlp_data_test.csv'

# ── Setup device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load state dict ──
state_dict = load_file(STATE_PATH, device='cpu')

# ── Load & prep data ──
data = pd.read_csv(DATA_PATH).dropna(axis=1)
# Identify feature columns by name
start_idx = data.columns.get_loc('img_mean_intensity')

feature_cols = data.columns[start_idx:]

input_data = data[feature_cols].values.astype('float32')
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

labels = data['k'].values.astype('float32').reshape(-1,1)

# Pull file name
if 'id' in data.columns:
    filenames = data['id'].values


# ── Load your model ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model_classes import MLP

model = MLP.MLP(input_size=input_data.shape[1])
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()

# ── Run inference ──
results = []
with torch.no_grad():
    for i in tqdm(range(len(input_data)), desc='Processing images', unit=' image(s)'):
        sample = torch.tensor(input_data[i:i+1], dtype=torch.float32, device=device)
        
        permeability = float(labels[i])
        
        output = model(sample).cpu().numpy()[0][0]
        pred = float(np.exp(output))
        results.append({
            'filename': os.path.basename(filenames[i]) + '_crop.jpeg', # To match the CNN predictions to be joined to a master CSV later
            'permeability': permeability,
            'prediction': pred
        })

# Save to CSV to save_file specified at beginning of script
pd.options.display.float_format = '{:.6e}'.format
results_df = pd.DataFrame(results)
results_df.to_csv(save_file, index=False, float_format='%.6e')
