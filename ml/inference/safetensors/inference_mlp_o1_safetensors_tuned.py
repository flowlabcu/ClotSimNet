import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

# ── Adjust these paths as needed ──
STATE_PATH = '/home/josh/ClotSimNet-Models/tuned/mlp_o1/mlp_o1_tuned.safetensors'
DATA_PATH  = '/home/josh/clotsimnet/data/mlp_data_aug_vert_horiz_flip.csv'
save_file = 'mlp_o1_tuned_vert_horiz_preds.csv'
# '/home/josh/clotsimnet/data/mlp_data_test.csv'

# ── Setup device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load state dict ──
state_dict = load_file(STATE_PATH, device='cpu')

# ── Load & prep data ──
data = pd.read_csv(DATA_PATH).dropna(axis=1)
# Identify feature columns by name
start_idx = data.columns.get_loc('img_mean_intensity')
end_idx = data.columns.get_loc('img_kurtosis_intensity')

feature_cols = data.columns[start_idx:end_idx+1]

input_data = data[feature_cols].values.astype('float32')
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

labels = data['k'].values.astype('float32').reshape(-1,1)

# Pull file name
if 'id' in data.columns:
    filenames = data['id'].values


# ── Load model ──
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model_classes import MLP

# Path to hyperparameter JSON file
hyperparams_path = '/home/josh/ClotSimNet-Models/hyperparameters/best_params_mlp_o1.json'
with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
        
hyperparameters = hyperparams['hyperparameters']


model = MLP.MLP(
    input_size=input_data.shape[1],
    hidden_size=hyperparameters['hidden_size'],
    num_layers=hyperparameters['num_layers'],
    learning_rate=hyperparameters['lr'],
    weight_decay=hyperparameters['weight_decay'],
    scheduler_factor=hyperparameters['scheduler_factor'],
    scheduler_patience=hyperparameters['scheduler_patience'],
    scheduler_threshold=hyperparameters['scheduler_threshold']
    )
model.load_state_dict(state_dict, strict=True)
model.to(device).eval()

# ── Run inference ──
results = []
with torch.no_grad():
    for i in range(len(input_data)):
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
