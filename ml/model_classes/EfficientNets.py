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
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
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

# Dynamically construct the path based on the user's home directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from utils import export_cnn, test_report

torch.set_float32_matmul_precision('highest') # Options are medium, high, highest

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16

# For training on AMD GPUs, as bfloat16 isn't supported
# tensor_dtype = torch.float16

# EfficientNet B0 architecture
class ENetB0(L.LightningModule):
    def __init__(
        self, 
        learning_rate: float=1e-3, 
        weight_decay: float=1e-2, 
        kernel_size: int=3, 
        stride: int=1, 
        padding: int=0,
        use_weights: bool=True, 
        scheduler_factor: float=0.1,
        scheduler_patience: int=10,
        scheduler_threshold: float=1e-4
    ):
        '''
        Class to load the EfficientNet B0 architecture from PyTorch's torchvision library.

        Parameters:
            learning_rate (float): Learning rate. Defaults to 1e-3.
            weight_decay (float): Weight decay for AdamW optimizer. Defaults to 1e-2.
            kernel_size (int): Kernel size for convolutional filters. Defaults to 3.
            stride (int): Stride for convolutional filters. Defaults to 1.
            padding (int): Padding for convolutional filters. Defaults to 0.
            use_weights (bool): Whether or not to use pretrained weights from ImageNet. Defaults to True.
            scheduler_factor (float): Factor for ReduceLROnPlateau. Defaults to 0.1.
            scheduler_patience (int): Patience for ReduceLROnPlateau. Defaults to 10.
            scheduler_threshold (float): Threshold for ReduceLROnPlateau. Defaults to 1e-4.
            
        Methods:
            All of the methods within this class are subclassed from PyTorch Lightning's LightningModule. Their documentation for each of these methods can be found here: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        '''
        super().__init__()
        self.save_hyperparameters()
        
        # Transforms flag to perform image augmentations
        # self.use_transforms = use_transforms
        
        # Save predictions for later
        self.test_preds = []
        self.test_labels = []
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_weights = use_weights
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold

        if use_weights==True:
            self.model = models.efficientnet_b0(weights='DEFAULT')
        else:
            self.model = models.efficientnet_b0()
            
        # Modify first conv. layer to accept grayscale/1 channel inputs
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # Modify classifier for regression output
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, 1)  # Single output for regression
        
        # Create dictionary of metrics to track for training, validation, and testing
        self.train_metrics = torchmetrics.MetricCollection(
            {
                'MAE': torchmetrics.regression.MeanAbsoluteError(num_outputs=1),
                'RMSE': torchmetrics.regression.MeanSquaredError(squared=False, num_outputs=1),
                'MSE': torchmetrics.regression.MeanSquaredError(squared=True, num_outputs=1),
               'r_squared': torchmetrics.regression.R2Score() 
            },
            prefix='train_'
        )
        
        self.valid_metrics = self.train_metrics.clone(prefix='valid_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')
        
        
    def forward(self, x):
        return self.model(x).squeeze(-1) # Remove extra dimension

    # Reset metrics
    def on_train_epoch_start(self):
        self.train_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        x, y = batch # Get inputs and labels
        y_pred = self(x) # Forward pass (call model's forward method)
        
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True) # Log MSE loss for monitoring
        
        # Log metrics
        batch_values = self.train_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # Reset metrics
    def on_validation_epoch_start(self):
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Update validation metrics
        batch_values = self.valid_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    # Reset metrics
    def on_test_epoch_start(self):
        self.test_metrics.reset()
        
    def test_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Update validation metrics
        batch_values = self.test_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        
        # Store predictions and labels
        self.test_preds.append(y_pred.cpu())
        self.test_labels.append(y.cpu())
        
        return loss
    
    def on_test_epoch_end(self):
        self.test_preds = torch.cat([pred.float() for pred in self.test_preds], dim=0).numpy()
        self.test_labels = torch.cat([label.float() for label in self.test_labels], dim=0).numpy()

        # Log transform test predictions and labels
        self.test_preds = np.exp(self.test_preds)
        self.test_labels = np.exp(self.test_labels)

        # Store results for access after trainer.test()
        self.test_results = {'preds': self.test_preds, 'labels': self.test_labels}
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, threshold=self.scheduler_threshold)
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'}

# EfficientNet B3 architecture
class ENetB3(L.LightningModule):
    def __init__(
        self, 
        learning_rate: float=1e-3, 
        weight_decay: float=1e-2, 
        kernel_size: int=3, 
        stride: int=1, 
        padding: int=0,
        use_weights: bool=True, 
        scheduler_factor: float=0.1,
        scheduler_patience: int=10,
        scheduler_threshold: float=1e-4
    ):
        '''
        Class to load the EfficientNet B3 architecture from PyTorch's torchvision library.

        Parameters:
            learning_rate (float): Learning rate. Defaults to 1e-3.
            weight_decay (float): Weight decay for AdamW optimizer. Defaults to 1e-2.
            kernel_size (int): Kernel size for convolutional filters. Defaults to 3.
            stride (int): Stride for convolutional filters. Defaults to 1.
            padding (int): Padding for convolutional filters. Defaults to 0.
            use_weights (bool): Whether or not to use pretrained weights from ImageNet. Defaults to True.
            scheduler_factor (float): Factor for ReduceLROnPlateau. Defaults to 0.1.
            scheduler_patience (int): Patience for ReduceLROnPlateau. Defaults to 10.
            scheduler_threshold (float): Threshold for ReduceLROnPlateau. Defaults to 1e-4.
            
        Methods:
            All of the methods within this class are subclassed from PyTorch Lightning's LightningModule. Their documentation for each of these methods can be found here: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        '''
        super().__init__()
        self.save_hyperparameters()
        
        # Transforms flag to perform image augmentations
        # self.use_transforms = use_transforms
        
        # Save predictions for later
        self.test_preds = []
        self.test_labels = []
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_weights = use_weights
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        
        if use_weights==True:
            self.model = models.efficientnet_b3(weights='DEFAULT')
        else:
            self.model = models.efficientnet_b3()
            
        # Modify first conv. layer to accept grayscale/1 channel inputs
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=40, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # Modify classifier for regression output
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, 1)  # Single output for regression
        
        # Create dictionary of metrics to track for training, validation, and testing
        self.train_metrics = torchmetrics.MetricCollection(
            {
                'MAE': torchmetrics.regression.MeanAbsoluteError(num_outputs=1),
                'RMSE': torchmetrics.regression.MeanSquaredError(squared=False, num_outputs=1),
                'MSE': torchmetrics.regression.MeanSquaredError(squared=True, num_outputs=1),
               'r_squared': torchmetrics.regression.R2Score() 
            },
            prefix='train_'
        )
        
        self.valid_metrics = self.train_metrics.clone(prefix='valid_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')
        
        
    def forward(self, x):
        return self.model(x).squeeze(-1) # Remove extra dimension

    # Reset metrics
    def on_train_epoch_start(self):
        self.train_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        x, y = batch # Get inputs and labels
        y_pred = self(x) # Forward pass (call model's forward method)
        
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True) # Log MSE loss for monitoring
        
        # Log metrics
        batch_values = self.train_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # Reset metrics
    def on_validation_epoch_start(self):
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Update validation metrics
        batch_values = self.valid_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    # Reset metrics
    def on_test_epoch_start(self):
        self.test_metrics.reset()
        
    def test_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Update validation metrics
        batch_values = self.test_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        
        # Store predictions and labels
        self.test_preds.append(y_pred.cpu())
        self.test_labels.append(y.cpu())
        
        return loss
    
    def on_test_epoch_end(self):
        self.test_preds = torch.cat([pred.float() for pred in self.test_preds], dim=0).numpy()
        self.test_labels = torch.cat([label.float() for label in self.test_labels], dim=0).numpy()

        # Log transform test predictions and labels
        self.test_preds = np.exp(self.test_preds)
        self.test_labels = np.exp(self.test_labels)

        # Store results for access after trainer.test()
        self.test_results = {'preds': self.test_preds, 'labels': self.test_labels}
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, threshold=self.scheduler_threshold)
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'} 
        
# EfficientNet B7 architecture        
class ENetB7(L.LightningModule):
    def __init__(
        self, 
        learning_rate: float=1e-3, 
        weight_decay: float=1e-2, 
        kernel_size: int=3, 
        stride: int=1, 
        padding: int=0,
        use_weights: bool=True, 
        scheduler_factor: float=0.1,
        scheduler_patience: int=10,
        scheduler_threshold: float=1e-4 
    ):
        '''
        Class to load the EfficientNet B7 architecture from PyTorch's torchvision library.

        Parameters:
            learning_rate (float): Learning rate. Defaults to 1e-3.
            weight_decay (float): Weight decay for AdamW optimizer. Defaults to 1e-2.
            kernel_size (int): Kernel size for convolutional filters. Defaults to 3.
            stride (int): Stride for convolutional filters. Defaults to 1.
            padding (int): Padding for convolutional filters. Defaults to 0.
            use_weights (bool): Whether or not to use pretrained weights from ImageNet. Defaults to True.
            scheduler_factor (float): Factor for ReduceLROnPlateau. Defaults to 0.1.
            scheduler_patience (int): Patience for ReduceLROnPlateau. Defaults to 10.
            scheduler_threshold (float): Threshold for ReduceLROnPlateau. Defaults to 1e-4.
            
        Methods:
            All of the methods within this class are subclassed from PyTorch Lightning's LightningModule. Their documentation for each of these methods can be found here: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        '''
        super().__init__()
        self.save_hyperparameters()
        
        # Transforms flag to perform image augmentations
        # self.use_transforms = use_transforms
        
        # Save predictions for later
        self.test_preds = []
        self.test_labels = []
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_weights = True
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        
        if use_weights==True:
            self.model = models.efficientnet_b7(weights='DEFAULT')
        else:
            self.model = models.efficientnet_b7()
            
        # Modify first conv. layer to accept grayscale/1 channel inputs
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # Modify classifier for regression output
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, 1) # Single output for regression
        
        # Create dictionary of metrics to track for training, validation, and testing
        self.train_metrics = torchmetrics.MetricCollection(
            {
                'MAE': torchmetrics.regression.MeanAbsoluteError(num_outputs=1),
                'RMSE': torchmetrics.regression.MeanSquaredError(squared=False, num_outputs=1),
                'MSE': torchmetrics.regression.MeanSquaredError(squared=True, num_outputs=1),
               'r_squared': torchmetrics.regression.R2Score() 
            },
            prefix='train_'
        )
        
        self.valid_metrics = self.train_metrics.clone(prefix='valid_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')
        
        
    def forward(self, x):
        return self.model(x).squeeze(-1) # Remove extra dimension

    # Reset metrics
    def on_train_epoch_start(self):
        self.train_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        x, y = batch # Get inputs and labels
        y_pred = self(x) # Forward pass (call model's forward method)
        
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True) # Log MSE loss for monitoring
        
        # Log metrics
        batch_values = self.train_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # Reset metrics
    def on_validation_epoch_start(self):
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Update validation metrics
        batch_values = self.valid_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    # Reset metrics
    def on_test_epoch_start(self):
        self.test_metrics.reset()
        
    def test_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        # Update validation metrics
        batch_values = self.test_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True, sync_dist=True)
        
        # Store predictions and labels
        self.test_preds.append(y_pred.cpu())
        self.test_labels.append(y.cpu())
        
        return loss
    
    def on_test_epoch_end(self):
        self.test_preds = torch.cat([pred.float() for pred in self.test_preds], dim=0).numpy()
        self.test_labels = torch.cat([label.float() for label in self.test_labels], dim=0).numpy()

        # Log transform test predictions and labels
        self.test_preds = np.exp(self.test_preds)
        self.test_labels = np.exp(self.test_labels)

        # Store results for access after trainer.test()
        self.test_results = {'preds': self.test_preds, 'labels': self.test_labels}
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, threshold=self.scheduler_threshold)
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'}
  