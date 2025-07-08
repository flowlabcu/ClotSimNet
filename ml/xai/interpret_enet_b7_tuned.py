import lightning as L
import torch
import os
import json
from os import path
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.amp import autocast

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from utils import export_cnn, test_report
from model_classes import EfficientNets, ResNets
from safetensors import safe_open


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


def interpret_base_low(
    model_path: str,
    model_name: str,
    low_pack_path: str,
    formatted_name: str,
    hyperparams_path: str
    ):
    
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
        
    hyperparameters = hyperparams['hyperparameters']
    
    
    model = EfficientNets.ENetB7(
        learning_rate=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay'],
        kernel_size=hyperparameters['kernel_size'],
        stride=hyperparameters['stride'],
        padding=hyperparameters['padding'],
        use_weights=True,
        scheduler_factor=hyperparameters['scheduler_factor'],
        scheduler_patience=hyperparameters['scheduler_patience'],
        scheduler_threshold=hyperparameters['scheduler_threshold']
    )

    # Load weights
    load_safetensors(model, model_path)

    # Device and dtype setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bfloat = device.type == 'cuda'
    # Move model to device and cast
    model = model.to(device)
    if use_bfloat:
        model = model.to(torch.bfloat16)
    model.eval()
    print(f'Using device: {device}, dtype: {"bfloat16" if use_bfloat else "float32"}')

    # Preprocess image
    image = Image.open(low_pack_path).convert('L')
    transform_pipeline = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform_pipeline(image)  # [1, H, W]
    input_tensor = image_tensor.unsqueeze(0).to(device)
    if use_bfloat:
        input_tensor = input_tensor.to(torch.bfloat16)
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # Forward pass (no-grad for attribute target baseline)
    with torch.no_grad():
        _ = model(input_tensor)

    # Select target conv layer
    if model_name in ('ENetB0','ENetB3','ENetB7'):
        target_layer = model.model.features[-1]
    else:
        target_layer = model.model.layer4[-1]

    # Init LayerGradCam
    layer_gc = LayerGradCam(model, target_layer)

    # Compute Grad-CAM attributions under autocast
    if use_bfloat:
        with autocast('cuda', dtype=torch.bfloat16):
            attributions_gc = layer_gc.attribute(input_tensor)
    else:
        attributions_gc = layer_gc.attribute(input_tensor)

    # Collapse channels and upsample
    heatmap = attributions_gc.mean(dim=1, keepdim=True)
    heatmap_up = F.interpolate(
        heatmap.float(),
        size=input_tensor.shape[2:],
        mode='bilinear',
        align_corners=False
    )

    # Normalize
    heat = heatmap_up[0, 0].cpu().detach().numpy()
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Original image for overlay
    orig = image_tensor.squeeze(0).cpu().numpy()

    # Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(orig, cmap='gray')
    hm = plt.imshow(heat_norm, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM Heatmap: Tuned {formatted_name}", fontweight='bold')
    plt.axis('off')
    cbar = plt.colorbar(hm, fraction=0.046, pad=0.04)
    cbar.set_label('Attribution Intensity', rotation=270, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=2)
    for tick in cbar.ax.yaxis.get_ticklabels():
        tick.set_weight('bold')

    out_name = f'grad_cam_plots/small_pack_{model_name.lower()}_tuned.pdf'
    plt.savefig(out_name, format='pdf', bbox_inches='tight')
    print(f"Saved Grad-CAM heatmap to {out_name}")
    
def interpret_base_high(
    model_path: str,
    model_name: str,
    high_pack_path: str,
    formatted_name: str,
    hyperparams_path: str
    ):
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
        
    hyperparameters = hyperparams['hyperparameters']
    
    
    model = EfficientNets.ENetB7(
        learning_rate=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay'],
        kernel_size=hyperparameters['kernel_size'],
        stride=hyperparameters['stride'],
        padding=hyperparameters['padding'],
        use_weights=True,
        scheduler_factor=hyperparameters['scheduler_factor'],
        scheduler_patience=hyperparameters['scheduler_patience'],
        scheduler_threshold=hyperparameters['scheduler_threshold']
    )

    # Load weights
    load_safetensors(model, model_path)

    # Device and dtype setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bfloat = device.type == 'cuda'
    # Move model to device and cast
    model = model.to(device)
    if use_bfloat:
        model = model.to(torch.bfloat16)
    model.eval()
    print(f'Using device: {device}, dtype: {"bfloat16" if use_bfloat else "float32"}')

    # Preprocess image
    image = Image.open(high_pack_path).convert('L')
    transform_pipeline = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform_pipeline(image)  # [1, H, W]
    input_tensor = image_tensor.unsqueeze(0).to(device)
    if use_bfloat:
        input_tensor = input_tensor.to(torch.bfloat16)
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # Forward pass (no-grad for attribute target baseline)
    with torch.no_grad():
        _ = model(input_tensor)

    # Select target conv layer
    if model_name in ('ENetB0','ENetB3','ENetB7'):
        target_layer = model.model.features[-1]
    else:
        target_layer = model.model.layer4[-1]

    # Init LayerGradCam
    layer_gc = LayerGradCam(model, target_layer)

    # Compute Grad-CAM attributions under autocast
    if use_bfloat:
        with autocast('cuda', dtype=torch.bfloat16):
            attributions_gc = layer_gc.attribute(input_tensor)
    else:
        attributions_gc = layer_gc.attribute(input_tensor)

    # Collapse channels and upsample
    heatmap = attributions_gc.mean(dim=1, keepdim=True)
    heatmap_up = F.interpolate(
        heatmap.float(),
        size=input_tensor.shape[2:],
        mode='bilinear',
        align_corners=False
    )

    # Normalize
    heat = heatmap_up[0, 0].cpu().detach().numpy()
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Original image for overlay
    orig = image_tensor.squeeze(0).cpu().numpy()

    # Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(orig, cmap='gray')
    hm = plt.imshow(heat_norm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    plt.title(f"Grad-CAM Heatmap: Tuned {formatted_name}", fontweight='bold')
    plt.axis('off')
    cbar = plt.colorbar(hm, fraction=0.046, pad=0.04)
    cbar.set_label('Attribution Intensity', rotation=270, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=2)
    for tick in cbar.ax.yaxis.get_ticklabels():
        tick.set_weight('bold')

    out_name = f'grad_cam_plots/large_pack_{model_name.lower()}_tuned.pdf'
    plt.savefig(out_name, format='pdf', bbox_inches='tight')
    print(f"Saved Grad-CAM heatmap to {out_name}")


if __name__ == '__main__':
    # Low packing fraction base models
    low_pack_frac_path = '/home/josh/clotsimnet/data/cnn_data_test_center_crop/cnn_data_center_crop/aN_447_rp_01700_seed_500_center_crop.jpeg'
    high_pack_frac_path = '/home/josh/clotsimnet/data/cnn_data_test_center_crop/cnn_data_center_crop/aN_509_rp_02400_seed_504_center_crop.jpeg'
    
    interpret_base_low(
        model_path='/home/josh/ClotSimNet-Models/tuned/enet_b7/enet_b7_tuned.safetensors',
        model_name='ENetB7',
        low_pack_path=low_pack_frac_path,
        formatted_name='EfficientNet-B7',
        hyperparams_path='/home/josh/ClotSimNet-Models/hyperparameters/best_params_enet_b7.json'
    )
    
    
    # High packing fraction base models
    interpret_base_high(
        model_path='/home/josh/ClotSimNet-Models/tuned/enet_b7/enet_b7_tuned.safetensors',
        model_name='ENetB7',
        high_pack_path=high_pack_frac_path,
        formatted_name='EfficientNet-B7',
        hyperparams_path='/home/josh/ClotSimNet-Models/hyperparameters/best_params_enet_b7.json'
    )