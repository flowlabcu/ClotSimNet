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

from utils import export_cnn, test_report
from model_classes import ResNets
from data_modules import image_transformations, load_data

"""
train_resnet_152.py

Trains the ResNet-152 architecture using tuned hyperparameters.

Things to change/adjust:

- lmdb_path (str): Path to LMDB data for training.
- model_type (str): Name of the model for WandB logging in hyperparameters.
- dataset_size (str): Size of the dataset.
- num_workers (int): Number of workers for parallel processing.
- max_epochs (int): Number of epochs to train for.
- hp_tune (str): Tuned vs base model for WandB logging.
- PROJECT_NAME (str): Name of the highest-level WandB project to log this run under.
- RUN_NAME (str): Name of the run that will pop up on WandB dashboard.
- model_name (str): Name of the exported model.
- model_dir (str): Path to directory to store the model
- hyperparams_path (str): Path to hyperparameter JSON file. Automatically looks for it in path clotsimnet/ml/hp_tune/hyperparams/best_params_resnet_152.json, but can be changed as needed.

Usage:
    python3 train_resnet_152.py

Returns:
    None
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
    
    lmdb_path = '/scratch/local/jogr4852_dir/cnn_data_5k_crop.lmdb'
    # lmdb_path = '/home/josh/clotsimnet/data_1k.lmdb'
    # lmdb_path = '/home/josh/clotsimnet/data_250.lmdb'
    
    # Hyperparams for logging
    model_type = 'Resnet152_tuned'
    dataset_size = '5k'
    
    num_workers = 70
    max_epochs = 500
     
    seed_everything(seed=seed_value, workers=True)
    
    transform_train = image_transformations.transform_train()
    transform_val = image_transformations.transform_val()
    transform_test = transform_val
    
    # Start logger    
    hp_tune = 'tuned' # base vs. tuned model
    
    PROJECT_NAME = 'ClotSimNet'
    RUN_NAME = 'ResNet152_Tuned'
    
    # Model export settings/names
    model_name = 'resnet_152_tuned'
    model_dir = '/projects/jogr4852/clotsimnet/ml/models/resnet_152'
    # model_dir = '/home/josh/clotsimnet/ml/models/resnet_50'
     
    # Path to hyperparameter JSON file
    hyperparams_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')), 'hp_tune', 'hyper_params', 'best_params_resnet_152.json')
    
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
        
    hyperparameters = hyperparams['hyperparameters']
    
    
    # TODO: Modify classes to pass scheduler factor, etc. in class definition
    model = ResNets.ResNet152(
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
    
    data_module = load_data.LMDBClotDataModule(
        lmdb_path=lmdb_path,
        batch_size=hyperparameters['batch_size'],
        num_workers=num_workers,
        train_transforms=transform_train,
        val_transforms=transform_val,
        test_transforms=transform_test
    )

    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
        
    
    wandb_logger = WandbLogger(
        project=PROJECT_NAME, 
        name=RUN_NAME
    )

    wandb_logger.log_hyperparams({
        'model': model_type,
        'dataset_size': dataset_size,
        'kernel_size': hyperparameters['kernel_size'],
        'stride': hyperparameters['stride'],
        'padding': hyperparameters['padding'],
        'num_workers': num_workers,
        'batch_size': hyperparameters['batch_size'],
        'max_epochs': max_epochs
    })

    # Train model
    trainer = Trainer(
        precision="bf16-mixed",
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=wandb_logger, # Comment out to remove logging
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=75, verbose=True)]
        )

    # Run training and testing
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, data_module)
    
    # Extract 5 of the model's predictions and labels
    preds = model.test_results['preds']
    labels = model.test_results['labels']
    
    # Create the nicely formatted test report
    test_report.create_report(preds=preds, labels=labels)

    # Export the trained model
    export_cnn.export(
        model=model,
        model_name=model_name,
        model_dir=model_dir
    )
        
