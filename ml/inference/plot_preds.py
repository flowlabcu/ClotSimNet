import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions(
    preds_csv: str,
    model_name: str,
    formatted_name: str,
    save_name: str
    ):
    
    weight='bold'
    font_size = 14

    df = pd.read_csv(preds_csv)
    
    permeability = df['permeability']
    prediction = df[model_name]
    
    # Want 5 examples that span the entire permeability range
    indices = np.linspace(start=0, stop=len(df)-1, num=5, dtype=int)
    
    permeability_subset = permeability.iloc[indices]
    prediction_subset = prediction.iloc[indices]
    
    # Bar width and positions
    bar_width = 0.35
    index = np.arange(5)  # Indices for the first five rows

    # Create the first plot: Permeability vs MLP Full Base and Tuned
    # plt.figure(figsize=(8, 6))
    plt.figure()
    plt.bar(index - bar_width / 2, permeability_subset, bar_width, label='Label', edgecolor='black')
    plt.bar(index + bar_width / 2, prediction_subset, bar_width, label=formatted_name, edgecolor='black')
    # plt.bar(index + bar_width / 2, mlp_full_tuned_subset, bar_width, label='MLP Full Tuned', edgecolor='black')

    # Customize the first plot
    plt.xlabel('Test instance', fontweight=weight, fontsize=font_size)
    plt.ylabel('Clot permeability (mm$^2$)', fontweight=weight, fontsize=font_size)
    plt.title(f'Permeability vs. {formatted_name}', fontweight=weight, fontsize=font_size)
    plt.xticks(index, [str(i+1) for i in index], fontweight=weight, fontsize=font_size)  # Use indices as x-axis labels
    plt.yticks(fontweight=weight, fontsize=font_size)
    plt.legend(loc='best',prop={'size': font_size, 'weight': weight})
    # plt.ylim([0, 10e-5])
    plt.yticks(fontweight=weight, fontsize=font_size)

    # make the scientific‐notation offset text bold too
    ax = plt.gca()
    offset = ax.yaxis.get_offset_text()
    offset.set_fontweight(weight)
    offset.set_fontsize(font_size)
    
    plt.tight_layout()
    plt.savefig(save_name)


preds_csv = '/home/josh/clotsimnet/data/preds/combined_preds.csv'
    
### ------------------ Base model plotting ------------------ ###

plot_predictions(
    preds_csv=preds_csv,
    model_name='enet_b0_base',
    formatted_name='EfficientNet-B0 Base',
    save_name=f'prediction_plots/enet_b0_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='enet_b3_base',
    formatted_name='EfficientNet-B3 Base',
    save_name=f'prediction_plots/enet_b3_base_preds.svg')


plot_predictions(
    preds_csv=preds_csv,
    model_name='enet_b7_base',
    formatted_name='EfficientNet-B7 Base',
    save_name=f'prediction_plots/enet_b7_base_preds.svg')


plot_predictions(
    preds_csv=preds_csv,
    model_name='resnet_18_base',
    formatted_name='ResNet-18 Base',
    save_name=f'prediction_plots/resnet_18_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='resnet_50_base',
    formatted_name='ResNet-50 Base',
    save_name=f'prediction_plots/resnet_50_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='resnet_152_base',
    formatted_name='ResNet-152 Base',
    save_name=f'prediction_plots/resnet_152_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='convnext_tiny_base',
    formatted_name='ConvNeXt-Tiny Base',
    save_name=f'prediction_plots/convnext_tiny_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='mlp_o1_base',
    formatted_name='First-order MLP Base',
    save_name=f'prediction_plots/mlp_o1_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='mlp_full_base',
    formatted_name='Full MLP Base',
    save_name=f'prediction_plots/mlp_full_base_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='mlp_dend_base',
    formatted_name='Dend. MLP Base',
    save_name=f'prediction_plots/mlp_dend_base_preds.svg')

## ------------------ Tuned model plotting ------------------ ###

plot_predictions(
    preds_csv=preds_csv,
    model_name='enet_b0_tuned',
    formatted_name='EfficientNet-B0 Tuned',
    save_name=f'prediction_plots/enet_b0_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='enet_b3_tuned',
    formatted_name='EfficientNet-B3 Tuned',
    save_name=f'prediction_plots/enet_b3_tuned_preds.svg')


plot_predictions(
    preds_csv=preds_csv,
    model_name='enet_b7_tuned',
    formatted_name='EfficientNet-B7 Tuned',
    save_name=f'prediction_plots/enet_b7_tuned_preds.svg')


plot_predictions(
    preds_csv=preds_csv,
    model_name='resnet_18_tuned',
    formatted_name='ResNet-18 Tuned',
    save_name=f'prediction_plots/resnet_18_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='resnet_50_tuned',
    formatted_name='ResNet-50 Tuned',
    save_name=f'prediction_plots/resnet_50_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='resnet_152_tuned',
    formatted_name='ResNet-152 Tuned',
    save_name=f'prediction_plots/resnet_152_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='convnext_tiny_tuned',
    formatted_name='ConvNeXt-Tiny Tuned',
    save_name=f'prediction_plots/convnext_tiny_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='mlp_o1_tuned',
    formatted_name='First-order MLP Tuned',
    save_name=f'prediction_plots/mlp_o1_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='mlp_full_tuned',
    formatted_name='Full MLP Tuned',
    save_name=f'prediction_plots/mlp_full_tuned_preds.svg')

plot_predictions(
    preds_csv=preds_csv,
    model_name='mlp_dend_tuned',
    formatted_name='Dend. MLP Tuned',
    save_name=f'prediction_plots/mlp_dend_tuned_preds.svg')
