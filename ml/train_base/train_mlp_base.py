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

"""
train_mlp_base.py

Trains the base full MLP model. 

Overview:
    - Reads in the dataset stored in CSV format at csv_path (user set).
    - Set logging parameters like model_type and dataset_size for reference later (optional).
    - Set number of worker processes and maximum number of training epochs.
    - Load data augmentation transforms, set project name for WandB logging.
    - Set path to save the trained model. Directory will be created if it does not exist already.
    - Train and test model.
    - Note: Patience is set to 70 epochs in the training loop. This can be changed manually.
    
Usage:
    python3 train_mlp_base.py
"""

torch.set_float32_matmul_precision('highest') # Options are medium, high, highest

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16

# For training on AMD GPUs, as bfloat16 isn't supported
# tensor_dtype = torch.float16

if __name__ == '__main__':
    
    ### ----------------------- ###
    # Set things like file paths, seed values, etc. here
    
    
    # Seed everything for reproducibility
    seed_value = 42
    
    # csv_path = '/scratch/local/jogr4852_dir/mlp_data_5k.csv'
    csv_path = '/home/josh/clotsimnet/data/mlp_data_5k.csv'
    
    # Hyperparams for logging
    model_type = 'MLP_full_base'
    dataset_size = '5k'
    
    PROJECT_NAME = 'ClotSimNet'
    RUN_NAME = 'poop'

    # Model export settings/names
    model_name = 'mlp_full_base'
    # model_dir = '/projects/jogr4852/clotsimnet/ml/models/mlp'
    model_dir = '/home/josh/clotsimnet/ml/models/mlp'

    num_workers = 18
    max_epochs = 350
    
    seed_everything(seed=seed_value, workers=True)
    
    # Load dataset
    data_module = load_data.MLPClotDataset(
        csv_path=csv_path,
        num_workers=num_workers
    )
    
    data_module.setup()
    
    model = MLP.MLP(
        input_size=data_module.X.shape[1]
    )
    
    print(f'Input size: {data_module.X.shape[1]}')

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
