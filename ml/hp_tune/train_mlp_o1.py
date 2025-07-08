import os
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import piexif
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.io import decode_image, read_file, image
from torchvision.transforms import v2
import torchvision.models as models
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
import wandb
import lmdb
import msgpack
import sys
import torchvision
import json

# Dynamically construct the path based on the user's home directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from utils import export_mlp, test_report
from model_classes import MLP
from data_modules import image_transformations, load_data

torch.set_float32_matmul_precision('highest') # Options are medium, high, highest

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16

# For training on AMD GPUs, as bfloat16 isn't supported
# tensor_dtype = torch.float16

if __name__ == '__main__':
    
    ### ----------------------- ###
    # Set things like file paths, seed values, etc. here
    
    
    # Seed everything for reproducibility
    seed_value = 42
    
    csv_path = '/scratch/local/jogr4852_dir/mlp_data_5k.csv'
    # csv_path = '/home/josh/clotsimnet/data/data_5k/mlp_data_5k.csv'
    
    # Hyperparams for logging
    model_type = 'MLP_o1_tuned'
    dataset_size = '5k'

    # num_workers = 70
    num_workers = 70
    max_epochs = 350
    
    seed_everything(seed=seed_value, workers=True)
    
    PROJECT_NAME = 'ClotSimNet'
    RUN_NAME = 'MLP_o1_Tuned'

    # Model export settings/names
    model_name = 'mlp_o1_tuned'
    model_dir = '/projects/jogr4852/clotsimnet/ml/models/mlp'
    # model_dir = '/home/josh/clotsimnet/ml/models/mlp'
    
    # Path to hyperparameter JSON file
    hyperparams_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')), 'hp_tune', 'hyper_params', 'best_params_mlp_o1.json')

    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
        
    hyperparameters = hyperparams['hyperparameters']
    
    # Load dataset
    data_module = load_data.MLPClotDataseto1(
        csv_path=csv_path,
        num_workers=num_workers,
        batch_size=hyperparameters['batch_size'])
    
    data_module.setup()
    
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

    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")

    # Start logger    
    hp_tune = 'base' # base vs. tuned model
    
    wandb_logger = WandbLogger(
        project=PROJECT_NAME, 
        name=RUN_NAME
    )

    wandb_logger.log_hyperparams({
        'model': model_type,
        'dataset_size': dataset_size,
        'num_workers': num_workers,
        'max_epochs': max_epochs
    })
    
    # Train model
    trainer = Trainer(
        precision='bf16-mixed',
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=wandb_logger, # Comment out to remove logging
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=70, verbose=True)]
        )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
    # Save and export model
    preds = model.test_results['preds']
    labels = model.test_results['labels']
    
    test_report.create_report(preds=preds, labels=labels)
    
    export_mlp.export(
        model=model,
        model_name=model_name,
        model_dir=model_dir,
        input_size=model.input_size
    )