import json
import numpy as np
import pandas as pd
import os
from os import path
import sys
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

# Captum imports for model interpretability
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

# Grab MLP model and data class from the base training script
# Dynamically construct the path based on the user's home directory
base_path = os.path.expanduser('~/clotsimnet')

# Add the utils directory to sys.path
sys.path.append(os.path.join(base_path, 'ml', 'data_modules'))
# Dynamically construct the path based on the user's home directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))
from load_data import MLPClotDataset, MLPClotDataseto1
from model_classes import MLP
from safetensors import safe_open

# Set global font size to 14
plt.rcParams.update({'font.size': 14})
font_size = 24
weight = 'bold'

def short_label(name: str) -> str:
    """
    Turn a long feature name into a concise one:
      - drop common prefixes (“img_”, “glcm_”, “laws_”)
      - collapse tokens into “Feature (d=…, a=…)” for GLCM,
        or just the statistic name for img_*
    """
    # 1) image‐stats  
    if name.startswith("img_"):
        core = name.replace("img_", "").replace("_intensity", "")
        return core.capitalize()
    # 2) GLCM features  
    if name.startswith("glcm_"):
        parts = name.split("_")
        feat = parts[1].capitalize()             # contrast, homogeneity, etc.
        if feat=='Asm':
            feat='ASM'
        else:
            pass
        # find the dist=<n> and angle=<n>
        try:
            d_i = parts.index("dist") + 1
            a_i = parts.index("angle") + 1
            return f"{feat} ({parts[d_i]}, {parts[a_i]})"
        except ValueError:
            return feat
    # 3) Laws textures  
    if name.startswith("laws_"):
        core = name.replace("laws_", "")
        # drop stats suffixes
        for suf in ("_mean", "_std", "_energy"):
            core = core.replace(suf, "")
        return core.upper()
    # fallback: ellipsize   
    return (name[:10] + "...") if len(name) > 13 else name

def load_safetensors(model_obj, model_path):
    if path.isfile(model_path):
        print(f'Loading pre-trained model from: {model_path}')
        tensors = {}
        with safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model_obj.load_state_dict(tensors)
    else:
        print('Model state_dict not found')

# Path to hyperparameter JSON file
hyperparams_path = '/home/josh/ClotSimNet-Models/hyperparameters/best_params_mlp.json'

with open(hyperparams_path, 'r') as f:
    hyperparams = json.load(f)
    
hyperparameters = hyperparams['hyperparameters']

csv_path = '/home/josh/clotsimnet/data/mlp_data_test.csv'
batch_size = hyperparameters['batch_size']
num_workers = 18

# Load dataset
data_module = MLPClotDataset(csv_path=csv_path, batch_size=batch_size, num_workers=num_workers)
data_module.setup()

data = pd.read_csv(csv_path)

data = data.dropna(axis=1) # Remove NaN values
# Get feature columns
start_idx = data.columns.get_loc('img_mean_intensity')
feature_cols = data.columns[start_idx:]

input_data = data[feature_cols].values.astype('float32')
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

labels = data['k'].values.astype('float32').reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.3, random_state=0)

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).view(-1, 1).float()

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).view(-1, 1).float()

datasets = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=True)

# Extract names of the features for plotting later
feature_names = data[feature_cols].columns.tolist()


model = MLP.MLP(
        input_size=data_module.X.shape[1],
        hidden_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers'],
        learning_rate=hyperparameters['lr'],
        weight_decay=hyperparameters['weight_decay'],
        scheduler_factor=hyperparameters['scheduler_factor'],
        scheduler_patience=hyperparameters['scheduler_patience'],
        scheduler_threshold=hyperparameters['scheduler_threshold']
    )

SAFETENSORS_MODEL_PATH = '/home/josh/ClotSimNet-Models/tuned/mlp_full/mlp_full_tuned.safetensors'
load_safetensors(model_obj=model, model_path=SAFETENSORS_MODEL_PATH)

# Set model into evaluation mode
model.eval()

# Run attribution algorithms
dl = DeepLift(model)
gs = GradientShap(model)
fa = FeatureAblation(model)

dl_attr_test = dl.attribute(X_test)
gs_attr_test = gs.attribute(X_test, X_train)
fa_attr_test = fa.attribute(X_test)

# Visualize scores
# Top N features

# prepare attributions for visualization

x_axis_data = np.arange(X_test.shape[1])
x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))


dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

gs_attr_test_sum = gs_attr_test.detach().numpy().sum(0)
gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)


lin_weight = model.model[0].weight[0].detach().numpy()
y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)


# Step 1: Compute absolute values of attributions
dl_abs     = np.abs(dl_attr_test_norm_sum)
gs_abs     = np.abs(gs_attr_test_norm_sum)
fa_abs     = np.abs(fa_attr_test_norm_sum)

# Step 2: Combine the importance values for an overall ranking.
# Here, we take an average (you could also use sum or another aggregation).
combined_importance = (dl_abs + gs_abs + fa_abs) / 3.0

# Step 3: Get indices for the top N features based on the combined importance.
top_n = 10  # adjust this number to change the number of top features
sorted_indices = np.argsort(-combined_importance)  # negative sign to sort in descending order
top_indices = sorted_indices[:top_n]

# Step 4: Subset the attribution arrays and feature names to only include top features.
top_dl     = dl_attr_test_norm_sum[top_indices]
top_gs     = gs_attr_test_norm_sum[top_indices]
top_fa     = fa_attr_test_norm_sum[top_indices]
top_feature_names = [feature_names[i] for i in top_indices]

# Step 5: Create x-axis data for plotting.
x_axis_data = np.arange(top_n)
width = 0.25  # defines the width of each bar

# Step 6: Plotting the data using a grouped bar chart.
plt.figure(figsize=(20, 10))
plt.grid(True, alpha=0.4)
plt.bar(x_axis_data + width,    top_dl,    width, label='DeepLIFT',         alpha=0.8)
plt.bar(x_axis_data + 2*width,    top_gs,    width, label='GradientSHAP',     alpha=0.8)
plt.bar(x_axis_data + 3*width,    top_fa,    width, label='Feature Ablation', alpha=0.8)

# Adjust x-axis ticks & labels
short_labels = [short_label(n) for n in top_feature_names]
plt.xticks(x_axis_data + 2.5*width, short_labels, rotation=45, ha='right', fontweight=weight, fontsize=font_size)
plt.ylabel('Attributions', fontweight=weight, fontsize=font_size)
plt.title('Top {} Input Feature Importances Across Multiple Algorithms, Full Tuned MLP'.format(top_n), fontweight=weight, fontsize=font_size)
plt.legend(loc='best', fontsize=font_size, prop={'weight':weight})
plt.ylim([-0.5, 0.5])

# Set bold font for ticks
plt.tick_params(axis='both', which='major', labelsize=font_size, labelcolor='black', width=1.5)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight(weight)
plt.tight_layout()
# plt.show()
plt.savefig('mlp_full_tuned_attribs.svg')