import pandas as pd

"""
test_report.py

Creates a report after a model has made predictions on the test dataset. Allows for easy visualization of model performance.

Usage:
    Called by training scripts automatically, no need to call this by itself.
"""

def create_report(preds, labels):
    df = pd.DataFrame({'Prediction': preds.flatten(),
                       'Label': labels.flatten()})
    
    # Adjust formatting since permeabilities are small
    pd.options.display.float_format = "{:.3e}".format
    print('------------------------Testing Report------------------------')
    print(df.head())
    print('--------------------------------------------------------------')
