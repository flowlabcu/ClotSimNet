import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score


# Function to perform paired t-test and print results for each model
def perform_test(preds_csv, model_name):
    
    # Read in prediction CSV file
    df = pd.read_csv(preds_csv)
    
    # Extract relevant data fields for label and predictions
    permeability = df['permeability']
    model_prediction = df[model_name]
    
    # Define significance level
    alpha = 0.05
    
    # Calculate paired t-test between predictions and permeability
    t_statistic, p_value = ttest_rel(permeability, model_prediction)
    
    # Calculate mean difference between prediction and the label
    mean_difference = np.mean(np.abs(permeability - model_prediction))
    
    # Calculate R^2 score
    r2score = r2_score(permeability, model_prediction)
    
    # Format the output for neat printing
    print(f"Model: {model_name}")
    print("-" * (7 + len(model_name)))
    print(f"Mean Difference: {mean_difference}")
    print(f"t-statistic: {t_statistic:.8f}")
    print(f"p-value:     {p_value:.8f}")
    print(f'R2 score:    {r2score:.8f}')
    
    if p_value < alpha:
        print("Result: The difference between predictions and permeability is statistically significant.\n")
    else:
        print("Result: No statistically significant difference between predictions and permeability.\n")

# Print header for the testing report
print("Testing Report: Base Models")
print("=" * 50)

preds_csv = '/home/josh/clotsimnet/data/preds/combined_preds.csv'

perform_test(preds_csv=preds_csv, model_name='convnext_tiny_base')

perform_test(preds_csv=preds_csv, model_name='enet_b0_base')
perform_test(preds_csv=preds_csv, model_name='enet_b3_base')
perform_test(preds_csv=preds_csv, model_name='enet_b7_base')

perform_test(preds_csv=preds_csv, model_name='resnet_18_base')
perform_test(preds_csv=preds_csv, model_name='resnet_50_base')
perform_test(preds_csv=preds_csv, model_name='resnet_152_base')

perform_test(preds_csv=preds_csv, model_name='mlp_full_base')
perform_test(preds_csv=preds_csv, model_name='mlp_o1_base')
perform_test(preds_csv=preds_csv, model_name='mlp_dend_base')

print("Testing Report: Tuned Models")
print("=" * 50)

perform_test(preds_csv=preds_csv, model_name='convnext_tiny_tuned')

perform_test(preds_csv=preds_csv, model_name='enet_b0_tuned')
perform_test(preds_csv=preds_csv, model_name='enet_b3_tuned')
perform_test(preds_csv=preds_csv, model_name='enet_b7_tuned')

perform_test(preds_csv=preds_csv, model_name='resnet_18_tuned')
perform_test(preds_csv=preds_csv, model_name='resnet_50_tuned')
perform_test(preds_csv=preds_csv, model_name='resnet_152_tuned')

perform_test(preds_csv=preds_csv, model_name='mlp_full_tuned')
perform_test(preds_csv=preds_csv, model_name='mlp_o1_tuned')
perform_test(preds_csv=preds_csv, model_name='mlp_dend_tuned')

print("Testing Report: MLP Data Aug")
print("=" * 50)

perform_test(preds_csv=preds_csv, model_name='mlp_full_tuned_vert_horiz')
perform_test(preds_csv=preds_csv, model_name='mlp_full_tuned_cnn_pipe')

perform_test(preds_csv=preds_csv, model_name='mlp_o1_tuned_vert_horiz')
perform_test(preds_csv=preds_csv, model_name='mlp_o1_tuned_cnn_pipe')

perform_test(preds_csv=preds_csv, model_name='mlp_dend_tuned_vert_horiz')
perform_test(preds_csv=preds_csv, model_name='mlp_dend_tuned_cnn_pipe')

perform_test(preds_csv=preds_csv, model_name='mlp_full_base_vert_horiz')
perform_test(preds_csv=preds_csv, model_name='mlp_full_base_cnn_pipe')

perform_test(preds_csv=preds_csv, model_name='mlp_o1_base_vert_horiz')
perform_test(preds_csv=preds_csv, model_name='mlp_o1_base_cnn_pipe')

perform_test(preds_csv=preds_csv, model_name='mlp_dend_base_vert_horiz')
perform_test(preds_csv=preds_csv, model_name='mlp_dend_base_cnn_pipe')
