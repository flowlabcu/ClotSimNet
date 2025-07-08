import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = [5.829e-05, 1.474e-05, 1.013e-05, 2.754e-05, 7.889e-06]

resnet_152 = [5.829e-05, 1.474e-05, 9.516e-06, 2.754e-05, 7.889e-06]

enet_b3 = [5.829e-05, 1.385e-05, 1.078e-05, 2.754e-05, 8.940e-06]

# Bar width and positions
bar_width = 0.35
index = np.arange(5)  # Indices for the first five rows

# Create the first plot: Permeability vs MLP Full Base and Tuned
plt.figure(figsize=(10, 6))
plt.bar(index - bar_width / 2, labels, bar_width, label='Label', edgecolor='black')
plt.bar(index + bar_width / 2, enet_b3, bar_width, label='EfficientNetB3', edgecolor='black')
# plt.bar(index + bar_width / 2, mlp_full_tuned_subset, bar_width, label='MLP Full Tuned', edgecolor='black')

# Customize the first plot
plt.xlabel('Test instance', fontweight='bold', fontsize=14)
plt.ylabel('Clot permeability', fontweight='bold', fontsize=14)
plt.title('Permeability vs. EfficientNetB3', fontweight='bold', fontsize=16)
plt.xticks(index, [str(i+1) for i in index], fontweight='bold')  # Use indices as x-axis labels
plt.yticks(fontweight='bold')
plt.legend(prop={'weight': 'bold'})
plt.ylim([0, 7e-5])
plt.tight_layout()
plt.savefig('enet_b3_base_preds.svg')
# plt.show()