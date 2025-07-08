import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss(loss_csv: str, save_path: str, title: str):
    """
    Plots the training and validation loss for models. Plots epochs on the x-axis, and log of loss on the y-axis (MSE loss).
    
    Parameters:
        loss_csv (str): Path to the CSV file containing epochs in column 0 and training/validation losses in columns 4 and 7, respectively. Pulled these from default loss curves from Weights & Biases (WandB) plots.
        save_path (str): Path to save the loss curve to.
        title (str): Title for the plot.
    """
    plt.figure()
    weight='bold'
    font_size = 14
    
    data = pd.read_csv(loss_csv)
    
    epoch = data.iloc[:, 0]
    train_loss = data.iloc[:, 4]
    val_loss = data.iloc[:, 7]
    
    plt.plot(epoch, train_loss, label='Train', linewidth=1.75)
    plt.plot(epoch, val_loss, label='Validation', linestyle='--', linewidth=1.5)
    plt.xlabel('Epoch', fontweight=weight, fontsize=font_size)
    plt.ylabel('MSE Loss', fontweight=weight, fontsize=font_size)
    plt.xticks(fontweight=weight, fontsize=font_size)
    plt.yticks(fontweight=weight, fontsize=font_size)
    plt.title(title, fontweight=weight, fontsize=font_size)
    plt.legend(loc='best',prop={'size': font_size, 'weight': weight})
    plt.grid(True)
    plt.ylim(10e-5, 10)
    plt.xlim(-10, 350)
    
    # Plot log-log
    # plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)

## Plot base models ###
plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_full_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_full_base_loss.svg', 
    title='Full MLP Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_o1_base_loss.csv',
    save_path='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_o1_base_loss.svg', 
    title='First-Order MLP Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_dend_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_dend_base_loss.svg', 
    title='Dendrogram MLP Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/convnext_tiny_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/convnext_tiny_base_loss.svg', 
    title='ConvNeXt-Tiny Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b0_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b0_base_loss.svg', 
    title='EfficientNet-B0 Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b3_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b3_base_loss.svg', 
    title='EfficientNet-B3 Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b7_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b7_base_loss.svg', 
    title='EfficientNet-B7 Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_18_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_18_base_loss.svg', 
    title='ResNet-18 Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_50_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_50_base_loss.svg', 
    title='ResNet-50 Base Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_152_base_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_152_base_loss.svg', 
    title='ResNet-152 Base Model'
)


## Plot tuned models ###
plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_full_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_full_tuned_loss.svg', 
    title='Full MLP Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_o1_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_o1_tuned_loss.svg', 
    title='First-Order MLP Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_dend_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/mlp_data/mlp_dend_tuned_loss.svg', 
    title='Dendrogram MLP Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/convnext_tiny_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/convnext_tiny_tuned_loss.svg', 
    title='ConvNeXt-Tiny Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b0_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b0_tuned_loss.svg', 
    title='EfficientNet-B0 Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b3_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b3_tuned_loss.svg',
    title='EfficientNet-B3 Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b7_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/enet_b7_tuned_loss.svg', 
    title='EfficientNet-B7 Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_18_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_18_tuned_loss.svg', 
    title='ResNet-18 Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_50_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_50_tuned_loss.svg', 
    title='ResNet-50 Tuned Model'
)

plot_loss(
    loss_csv='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_152_tuned_loss.csv', 
    save_path='/home/josh/clotsimnet/figures/loss_curves/cnn_data/resnet_152_tuned_loss.svg', 
    title='ResNet-152 Tuned Model'
)