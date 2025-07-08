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

test_csv_path = '/home/josh/clotsimnet/data/test_dataset/mlp_data_test_dataset.csv'
batch_size = 16
num_workers = 18

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


def mlp_full_base(model_path, test_csv_path):
    batch_size = 16
    num_workers = 18
    data_module = MLPClotDataset(csv_path=test_csv_path, batch_size=batch_size, num_workers=num_workers)
    data_module.setup()
    
    model = MLP.MLP(
    input_size=data_module.X.shape[1]
    )
    
    load_safetensors(model_obj=model, model_path=model_path)
    
    model.eval()
    
    dl = DeepLift(model)
    gs = GradientShap(model)
    fa = FeatureAblation(model)

    dl_attr_test = dl.attribute(X_test)
    gs_attr_test = gs.attribute(X_test, X_train)
    fa_attr_test = fa.attribute(X_test)
    
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
    width = 0.14  # defines the width of each bar

    # Step 6: Plotting the data using a grouped bar chart.
    plt.figure(figsize=(20, 10))
    plt.bar(x_axis_data + width,    top_dl,    width, label='DeepLift',         alpha=0.8, color='#34b8e0')
    plt.bar(x_axis_data + 1*width,    top_gs,    width, label='GradientShap',     alpha=0.8, color='#4260f5')
    plt.bar(x_axis_data + 2*width,    top_fa,    width, label='Feature Ablation', alpha=0.8, color='#49ba81')

    # Adjust x-axis ticks & labels
    plt.xticks(x_axis_data + 2.5*width, top_feature_names, rotation=45, ha='right', fontweight='bold')
    plt.ylabel('Normalized Attributions', fontweight='bold')
    plt.title('Top {} Input Feature Importances Across Multiple Algorithms, Full Base MLP'.format(top_n), fontweight='bold')
    plt.legend(loc='best', prop={'weight':'bold'})

    # Set bold font for ticks
    plt.tick_params(axis='both', which='major', labelsize=12, labelcolor='black', width=1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('test_attribs.svg')

def mlp_full_tuned(model_path, test_csv_path, hyperparams_path):
    
    num_workers = 18
    
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)

    hyperparameters = hyperparams['hyperparameters']
    
    batch_size = hyperparameters['batch_size']
    
    data_module = MLPClotDataset(csv_path=test_csv_path, batch_size=batch_size, num_workers=num_workers)
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
    
    
    # Create MLP with tuned hyperparameters
    model = MLP.MLP(
        input_size=data_module.X.shape[1],
        hidden_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers'],
        learning_rate=hyperparameters['lr'],
        weight_decay=hyperparameters['weight_decay'],
        first_bias=hyperparameters['first_bias'],
        last_bias=hyperparameters['last_bias'],
        scheduler_factor=hyperparameters['scheduler_factor'],
        scheduler_patience=hyperparameters['scheduler_patience'],
        scheduler_threshold=hyperparameters['scheduler_threshold']
    )
    
    # Load weights
    load_safetensors(model_obj=model, model_path=model_path)
    
    # Set model into evaluation mode
    model.eval()
    
    # ig = IntegratedGradients(model)
    # ig_nt = NoiseTunnel(ig)
    dl = DeepLift(model)
    gs = GradientShap(model)
    fa = FeatureAblation(model)

    # ig_attr_test = ig.attribute(X_test, n_steps=50)
    # ig_nt_attr_test = ig_nt.attribute(X_test)
    dl_attr_test = dl.attribute(X_test)
    gs_attr_test = gs.attribute(X_test, X_train)
    fa_attr_test = fa.attribute(X_test)
    
    # Visualize top n attribution scores
    x_axis_data = np.arange(X_test.shape[1])
    x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

    # ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
    # ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

    # ig_nt_attr_test_sum = ig_nt_attr_test.detach().numpy().sum(0)
    # ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

    dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    gs_attr_test_sum = gs_attr_test.detach().numpy().sum(0)
    gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

    fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
    fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)


    lin_weight = model.model[0].weight[0].detach().numpy()
    y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)


    # Step 1: Compute absolute values of attributions
    ig_abs     = np.abs(ig_attr_test_norm_sum)
    ig_nt_abs  = np.abs(ig_nt_attr_test_norm_sum)
    dl_abs     = np.abs(dl_attr_test_norm_sum)
    gs_abs     = np.abs(gs_attr_test_norm_sum)
    fa_abs     = np.abs(fa_attr_test_norm_sum)
    w_abs      = np.abs(y_axis_lin_weight)

    # Step 2: Combine the importance values for an overall ranking.
    # Here, we take an average (you could also use sum or another aggregation).
    combined_importance = (ig_abs + ig_nt_abs + dl_abs + gs_abs + fa_abs + w_abs) / 6.0

    # Step 3: Get indices for the top N features based on the combined importance.
    top_n = 10  # adjust this number to change the number of top features
    sorted_indices = np.argsort(-combined_importance)  # negative sign to sort in descending order
    top_indices = sorted_indices[:top_n]

    # Step 4: Subset the attribution arrays and feature names to only include top features.
    top_ig     = ig_attr_test_norm_sum[top_indices]
    top_ig_nt  = ig_nt_attr_test_norm_sum[top_indices]
    top_dl     = dl_attr_test_norm_sum[top_indices]
    top_gs     = gs_attr_test_norm_sum[top_indices]
    top_fa     = fa_attr_test_norm_sum[top_indices]
    top_w      = y_axis_lin_weight[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]

    # Step 5: Create x-axis data for plotting.
    x_axis_data = np.arange(top_n)
    width = 0.14  # defines the width of each bar

    # Step 6: Plotting the data using a grouped bar chart.
    plt.figure(figsize=(20, 10))
    # plt.bar(x_axis_data,             top_ig,    width, label='Int Grads',        alpha=0.8, color='#eb5e7c')
    # plt.bar(x_axis_data + 1*width,    top_ig_nt, width, label='Int Grads w/SmoothGrad', alpha=0.8, color='#A90000')
    plt.bar(x_axis_data + 2*width,    top_dl,    width, label='DeepLift',         alpha=0.8, color='#34b8e0')
    plt.bar(x_axis_data + 3*width,    top_gs,    width, label='GradientShap',     alpha=0.8, color='#4260f5')
    plt.bar(x_axis_data + 4*width,    top_fa,    width, label='Feature Ablation', alpha=0.8, color='#49ba81')
    # plt.bar(x_axis_data + 5*width,    top_w,     width, label='Weights',          alpha=0.8, color='grey')

    # Adjust x-axis ticks & labels
    plt.xticks(x_axis_data + 2.5*width, top_feature_names, rotation=45, ha='right', fontweight='bold')
    plt.ylabel('Normalized Attributions', fontweight='bold')
    plt.title('Top {} Input Feature Importances Across Multiple Algorithms, Full Tuned MLP'.format(top_n), fontweight='bold')
    plt.legend(loc='best', prop={'weight':'bold'})

    # Set bold font for ticks
    plt.tick_params(axis='both', which='major', labelsize=12, labelcolor='black', width=1.5)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.show()

test_csv_path = '/home/josh/clotsimnet/data/mlp_data_test.csv'

mlp_full_base(model_path='/home/josh/clotsimnet/ml/models/mlp/mlp_full_base_inc.safetensors', test_csv_path=test_csv_path)

# mlp_full_tuned(model_path='/home/josh/clotsimnet/ml/models/mlp/mlp_full_tuned_inc2.safetensors')