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
    
    
    model = EfficientNets.ENetB0(
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
    # Move model to device and cast
    model = model.to(device)
    model.eval()
    print(f'Using device: {device}')

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
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # Forward pass (no-grad for attribute target baseline)
    with torch.no_grad():
        _ = model(input_tensor)

    # Select target conv layer
    if model_name in ('ENetB0','ENetB3','ENetB7'):
        target_layer = model.model.features[-1]
    else:
        target_layer = model.model.layer4[-1]

    # Initialize LayerGradCam
    layer_gc = LayerGradCam(model, target_layer)

    # Compute Grad-CAM attributions
    attributions_gc = layer_gc.attribute(input_tensor)

    # Upsample Grad-CAM headmap to match the input size
    attributions_gc_upsampled = F.interpolate(attributions_gc,
                                            size=input_tensor.shape[2:],
                                            mode='bilinear')

    # Convert to numpy for visualization
    attributions_gc_np = attributions_gc_upsampled.squeeze().detach().cpu().numpy()

    # Normalize
    attributions_gc_norm = (attributions_gc_np - attributions_gc_np.min()) / (attributions_gc_np.max() - attributions_gc_np.min())

    image_np = input_tensor[0, 0].cpu().numpy()

    # Visualize the original image with the Grad-CAM heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np, cmap="gray")
    heatmap = plt.imshow(attributions_gc_norm, cmap='jet', alpha=0.5)
    # plt.imshow(attributions_gc_norm, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap: Base EfficientNet-B0", fontweight='bold')
    plt.axis("off")
    # Add a color bar as a legend for the heatmap
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Attribution Intensity", rotation=270, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=2)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight('bold')

    out_name = f'grad_cam_plots/small_pack_{model_name.lower()}_tuned_v2.pdf'
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
    # Move model to device and cast
    model = model.to(device)
    model.eval()
    print(f'Using device: {device}')

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
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # Forward pass (no-grad for attribute target baseline)
    with torch.no_grad():
        _ = model(input_tensor)

    # Select target conv layer
    if model_name in ('ENetB0','ENetB3','ENetB7'):
        target_layer = model.model.features[-1]
    else:
        target_layer = model.model.layer4[-1]

    # Initialize LayerGradCam
    layer_gc = LayerGradCam(model, target_layer)

    # Compute Grad-CAM attributions
    attributions_gc = layer_gc.attribute(input_tensor)

    # Upsample Grad-CAM headmap to match the input size
    attributions_gc_upsampled = F.interpolate(attributions_gc,
                                            size=input_tensor.shape[2:],
                                            mode='bilinear')

    # Convert to numpy for visualization
    attributions_gc_np = attributions_gc_upsampled.squeeze().detach().cpu().numpy()

    # Normalize
    attributions_gc_norm = (attributions_gc_np - attributions_gc_np.min()) / (attributions_gc_np.max() - attributions_gc_np.min())

    image_np = input_tensor[0, 0].cpu().numpy()

    # Visualize the original image with the Grad-CAM heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np, cmap="gray")
    heatmap = plt.imshow(attributions_gc_norm, cmap='jet', alpha=0.5)
    # plt.imshow(attributions_gc_norm, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap: Base EfficientNet-B0", fontweight='bold')
    plt.axis("off")
    # Add a color bar as a legend for the heatmap
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Attribution Intensity", rotation=270, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=2)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight('bold')

    out_name = f'grad_cam_plots/large_pack_{model_name.lower()}_tuned_v2.pdf'
    plt.savefig(out_name, format='pdf', bbox_inches='tight')
    print(f"Saved Grad-CAM heatmap to {out_name}")


if __name__ == '__main__':
    # Low packing fraction base models
    low_pack_frac_path = '/projects/jogr4852/clotsimnet/data/cnn_data_test_center_crop/cnn_data_center_crop/aN_447_rp_01700_seed_500_center_crop.jpeg'
    high_pack_frac_path = '/projects/jogr4852/clotsimnet/data/cnn_data_test_center_crop/cnn_data_center_crop/aN_509_rp_02400_seed_504_center_crop.jpeg'
    
    interpret_base_low(
        model_path='/projects/jogr4852/clotsimnet/models/tuned/enet_b7/enet_b7_tuned.safetensors',
        model_name='ENetB7',
        low_pack_path=low_pack_frac_path,
        formatted_name='EfficientNet-B7',
        hyperparams_path='/projects/jogr4852/clotsimnet/models/hyperparameters/best_params_enet_b7.json'
    )
    
    
    # High packing fraction base models
    interpret_base_high(
        model_path='/projects/jogr4852/clotsimnet/models/tuned/enet_b7/enet_b7_tuned.safetensors',
        model_name='ENetB7',
        high_pack_path=high_pack_frac_path,
        formatted_name='EfficientNet-B7',
        hyperparams_path='/projects/jogr4852/clotsimnet/models/hyperparameters/best_params_enet_b7.json'
    )