import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_excel('/home/josh/clotsimnet/ml/inference/mlp_preds.xlsx')

# Extract relevant columns (adjust indices as needed)
permeability = df.iloc[:, 0]
mlp_full_tuned = df.iloc[:, 2]

# Select the first five rows for plotting
permeability_subset = permeability[:5]
mlp_full_tuned_subset = mlp_full_tuned[:5]

# Bar width and positions
bar_width = 0.35
index = np.arange(5)

# Create two subplots (upper and lower) with shared x-axis.
# Height ratios are set so the top subplot gets more space.
fig, (ax_top, ax_bottom) = plt.subplots(2, 1,
                                        sharex=True,
                                        figsize=(10, 6),
                                        gridspec_kw={'height_ratios': [3, 1]})

# Plot the data on both axes (using the same colors for consistency)
ax_top.bar(index - bar_width / 2, permeability_subset, bar_width,
           label='Label', edgecolor='black', color='tab:blue')
ax_top.bar(index + bar_width / 2, mlp_full_tuned_subset, bar_width,
           label='Tuned Full MLP', edgecolor='black', color='tab:orange')

ax_bottom.bar(index - bar_width / 2, permeability_subset, bar_width,
              label='Label', edgecolor='black', color='tab:blue')
ax_bottom.bar(index + bar_width / 2, mlp_full_tuned_subset, bar_width,
              label='Tuned Full MLP', edgecolor='black', color='tab:orange')

# Define the break value (for example, here half of 0.00025)
break_val = 0.00025 / 2  # equals 0.000125

# Set y-axis limits on each subplot:
ax_bottom.set_ylim(0, break_val * 1.1)  # lower axis: around the small value range
ax_top.set_ylim(break_val * 1.3, max(mlp_full_tuned_subset) * 1.1)  # upper axis: remaining larger values

# Hide the shared boundary spines
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)

# Remove x-axis tick labels from the top subplot (kept only on the bottom)
ax_top.tick_params(labelbottom=False)

# Set custom x-ticks on the bottom axis and make the labels bold
ax_bottom.set_xticks(index)
ax_bottom.set_xticklabels([str(i+1) for i in index], fontweight='bold')

# Draw diagonal “break” lines on both subplots.
# These lines mimic the style from the Matplotlib documentation.
d = 0.015  # Size of diagonal lines in axis coordinates
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Center the common y-axis label vertically on the left side of the entire figure.
# The 'fig.text' command places text in figure coordinates.
fig.text(0.03, 0.5, 'Clot permeability',
         ha='center', va='center', rotation='vertical',
         fontweight='bold', fontsize=14)

# Set the x-axis label for the bottom subplot and the title for the top subplot
ax_bottom.set_xlabel('Test instance', fontweight='bold', fontsize=14)
ax_top.set_title('Permeability vs. Full Tuned MLP', fontweight='bold', fontsize=16)

# Add a legend on the top subplot with bold font
ax_top.legend(loc='upper right', prop={'weight': 'bold'})

# Ensure y-axis tick labels are bold on both subplots
for tick in ax_top.get_yticklabels():
    tick.set_fontweight('bold')
for tick in ax_bottom.get_yticklabels():
    tick.set_fontweight('bold')

# Adjust vertical spacing between the subplots
plt.subplots_adjust(hspace=0.05)
# plt.tight_layout()
# plt.show()

plt.savefig('mlp_full_tuned_preds.svg')
