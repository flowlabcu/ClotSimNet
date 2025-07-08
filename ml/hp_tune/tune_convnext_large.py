import os
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics
import wandb
import lmdb
import msgpack
import sys
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from ray.air.integrations.wandb import WandbLoggerCallback
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
# from utils import export_mlp, test_report
import json

# Packages for hyperparameter tuning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))
from data_modules import image_transformations

"""
tune_convnext_large.py

This script tunes a ConvNeXt-Large model for predicting permeability from grayscale CFD images
stored in an LMDB. It uses Ray Tune with the ASHA scheduler to search over hyperparameters and 
leverages PyTorch Lightning and Weights & Biases (W&B) for scalable training and logging.

Key Features:
-------------
- Loads 512x512 grayscale images and log-transformed permeability labels from LMDB.
- Applies heavy data augmentation during training (flipping, rotation, affine transforms).
- Uses ConvNeXt-Large with a custom grayscale input stem and single regression output.
- Hyperparameter search over learning rate, weight decay, conv parameters, batch size, and scheduler settings.
- Tracks metrics (MAE, RMSE, MSE, R2) using TorchMetrics.
- Automatically saves best config and loss to `hyper_params/best_params_convnext_large.json`.

Required:
---------
- A prebuilt LMDB with `image` (bytes) and `permeability` (float) fields.
- W&B API key and login (or set `WANDB_API_KEY` env var).
- GPUs with bfloat16 or float16 support.

Usage:
------
Run the script as a standalone training/tuning job:
    python tune_convnext_large.py

If interrupted, it will attempt to restore from:
    CHECKPOINT_DIR/PROJECT_NAME

Output:
-------
- Best config and loss printed at the end.
- Best config saved as JSON.
- W&B logs each run under the `ray_tune_runs` group.

Notes: 
    - The image transformations in here should match those in /clotsimnet/ml/data_modules/image_transformations.py. I tried calling that module from here but was unable to get it to work. Sorry :(
    - Edit everything under the `training settings` comment as necessary, as well as DEFAULT_CONFIG and SEARCH_SPACE.
"""

torch.set_float32_matmul_precision('highest') # Options are medium, high, highest

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16

# For training on AMD GPUs, as bfloat16 isn't supported
# tensor_dtype = torch.float16

### ----- Define Search Space for Tuning ---- ###

DEFAULT_CONFIG = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'batch_size': 4,
    'scheduler_factor': 0.1,
    'scheduler_patience': 10,
    'scheduler_threshold': 1e-4,
}

SEARCH_SPACE = {
        'learning_rate': tune.loguniform(1e-6, 1e-1), 
        'weight_decay': tune.loguniform(1e-8, 1e-2),
        'kernel_size': tune.choice([1, 3, 5, 7]),
        'stride': tune.choice([1, 2]),
        'padding': tune.choice([0, 1, 2, 3,]),
        'batch_size': tune.choice([4, 8, 16, 32, 64, 128, 256]),
        'scheduler_factor': tune.loguniform(1e-5, 1e-1),
        'scheduler_patience': tune.choice([1, 2, 5, 10, 20, 40]),
        'scheduler_threshold': tune.loguniform(1e-5, 1e-1),       
}

# Training settings
NUM_EPOCHS = 25 # Number of training epochs
NUM_SAMPLES = 25 # Number of samples from parameter space
LMDB_PATH = '/scratch/local/jogr4852_dir/data_5k.lmdb'
# LMDB_PATH = '/home/josh/clotsimnet/data_5k.lmdb'
CHECKPOINT_DIR = '/scratch/alpine/jogr4852/ray_checkpoints/'
# CHECKPOINT_DIR = '/home/josh/ray_checkpoints/'
PROJECT_NAME = 'tune_convnext_large'

SCALING_CONFIG = ScalingConfig(
    num_workers=1, 
    use_gpu=True, 
    resources_per_worker={"GPU": 1, 'CPU': 64}
)

CHECKPOINT_CONFIG = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute='val_loss',
    checkpoint_score_order='min',
    checkpoint_frequency=0
)


class LMDBClotDataset(Dataset):
    # Class variable to store image directory
    _lmdb_path = None
    
    @classmethod
    def set_image_dir(
        cls, 
        lmdb_path
    ):
        '''
        Class method to set the LMDB directory once
        '''
        cls._lmdb_path = lmdb_path
        
    def __init__(
        self, 
        transform=None, 
        indices=None
    ):
        if self._lmdb_path is None:
            raise ValueError('Set lmdb_path using CNNClotDataset.set_lmdb_path() before creating instances')
        
        self.transform = transform
        self.env = lmdb.open(self._lmdb_path, readonly=True, lock=False)
        
        with self.env.begin(write=False) as txn:
            self.keys = list(txn.cursor().iternext(values=False)) # Get all keys/image names
            
        if indices is not None:
            self.keys = [self.keys[i] for i in indices if i < len(self.keys)]
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        with self.env.begin(write=False) as txn:
            data = msgpack.unpackb(txn.get(key), raw=False) # Load pickled data
        
        # Extract image bytes
        image_bytes = data['image']
        
        # Convert to tensor
        image_tensor = torchvision.io.decode_image(
            torch.tensor(bytearray(image_bytes), dtype=torch.uint8),
            mode=torchvision.io.image.ImageReadMode.GRAY).float()

        # Apply transformations
        if self.transform:
            image = self.transform(image_tensor)
        
        permeability = np.log(data['permeability']) # Log-transform the permeability
        permeability = torch.tensor(float(permeability), dtype=tensor_dtype)
            
        return image, permeability
    
    def close(self):
        '''
        Close LMDB dataset when not in use
        '''
        if hasattr(self, 'env'):
            self.env.close()
    
class LMDBClotDataModule(L.LightningDataModule):
    def __init__(
        self, 
        lmdb_path: str,
        config,
        num_workers: int=18,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        seed: int=42
    ):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.batch_size = config['batch_size']
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.seed = seed
        
        # Set LMDB path
        LMDBClotDataset.set_image_dir(lmdb_path)
        
        # Dataset splits
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
    def setup(self, stage: Optional[str]=None):
        
        # Training validation split, remainder is for the test set
        train_frac = 0.8
        val_frac = 0.1
        
        # Create temporary dataset
        temp_dataset = LMDBClotDataset()
        dataset_size = len(temp_dataset)
        temp_dataset.close()
        
        train_size = int(train_frac * dataset_size)
        val_size = int(val_frac * dataset_size)
        test_size = int(dataset_size - train_size - val_size)
        
        
        # Create indices for train/test/val splits
        indices = list(range(dataset_size))
        
        # Generate fixed seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        
        self.train_indices, self.val_test_indices = random_split(indices, [train_size, val_size + test_size], generator=generator)
        self.val_indices, self.test_indices = random_split(self.val_test_indices, [val_size, test_size], generator=generator)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = LMDBClotDataset(
                transform=self.train_transforms,
                indices=self.train_indices
            )
            self.val_dataset = LMDBClotDataset(
                transform=self.val_transforms,
                indices=self.val_indices
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = LMDBClotDataset(
                transform=self.test_transforms,
                indices=self.test_indices
            )
    
    # NOTE: pin_memory=True calls in the next three defs should be monitored for performance. Can easily set to false.
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True, 
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=True
        )
        
    def teardown(self, stage: Optional[str] = None):
        if hasattr(self, 'train_dataset'):
            self.train_dataset.close()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.close()
        if hasattr(self, 'test_dataset'):
            self.test_dataset.close()
    
# ConvNext-Large architecture
class ConvNextLarge(L.LightningModule):
    def __init__(
        self,
        config, 
        seed: int=42   
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Save predictions for later
        self.test_preds = []
        self.test_labels = []
        
        # Tuned hyperparameters
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.kernel_size = config['kernel_size']
        self.stride = config['stride']
        self.padding = config['padding']
        self.use_weights = True
        self.batch_size = config['batch_size']
        self.scheduler_factor = config['scheduler_factor']
        self.scheduler_patience = config['scheduler_patience']
        self.scheduler_threshold = config['scheduler_threshold']
        
        # Non-tuned hyperparameters
        # self.num_workers = num_workers
        self.seed = seed
        
        if self.use_weights==True:
            self.model = models.convnext_large(weights='DEFAULT')
        else:
            self.model = models.convnext_large()
            
        # Modify first conv. layer to accept grayscale/1 channel inputs
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=192, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
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
    
    
def train_func(config):
    
    # Create new WandBLogger for training run
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        reinit=True
    )
    
    # Log hyperparameters for this run
    wandb_logger.log_hyperparams({
        'learning_rate': config['learning_rate'], 
        'weight_decay': config['weight_decay'],
        'kernel_size': config['kernel_size'],
        'stride': config['stride'],
        'padding': config['padding'],
        'batch_size': config['batch_size'],
        'scheduler_factor': config['scheduler_factor'],
        'scheduler_patience': config['scheduler_patience'],
        'scheduler_threshold': config['scheduler_threshold']
    })
    
    # Seed everything for reproducibility
    seed_value = 42
    seed_everything(seed=seed_value, workers=True)
    
    # Define transforms
    transform_train = image.transformations.transform_train()

    # Validation/test transforms (only preprocessing, no data augmentation)
    transform_val = image_transformations.transform_val()
    transform_test = transform_val
    
    # Load dataset
    lmdb_path = LMDB_PATH
    LMDBClotDataset.set_image_dir(lmdb_path)
    
    num_workers = 64
    data_module = LMDBClotDataModule(
        lmdb_path=lmdb_path,
        num_workers=num_workers,
        config=config,
        train_transforms=transform_train,
        val_transforms=transform_val,
        test_transforms=transform_test
    )
    
    # Create model
    model = ConvNextLarge(
        config=config,
        seed=seed_value
    )
    
    # Create callback to log the validation loss at the end of training
    tune_callback = TuneReportCheckpointCallback(
        metrics={"val_loss": "val_loss"}, 
        on="fit_end",
    )
    
    # Train model
    trainer = Trainer(
        precision='bf16-mixed',
        accelerator='auto',
        devices='auto',
        strategy=RayDDPStrategy(),
        callbacks=[tune_callback],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        log_every_n_steps=10
        )
    
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
def tune_cnn(config, num_samples=NUM_SAMPLES, restore_path=None):
    scheduler = ASHAScheduler(max_t=NUM_EPOCHS, grace_period=1, reduction_factor=2)
    
    # Create WandbLoggerCallback
    wandb_callback = WandbLoggerCallback(
        project=PROJECT_NAME,
        api_key=None,
        log_config=True,
        resume='allow',
        group='ray_tune_runs'
    )
    
    RUN_CONFIG = RunConfig(
    checkpoint_config=CHECKPOINT_CONFIG,
    callbacks=[wandb_callback],
    name=PROJECT_NAME,
    storage_path=CHECKPOINT_DIR
    )
    
    # Create Ray trainer with wandb
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=SCALING_CONFIG,
        run_config=RUN_CONFIG,
    )

    if restore_path and os.path.exists(restore_path) and os.listdir(restore_path):
        try:
            print(f'Restoring tuner from {restore_path}')
            tuner = tune.Tuner.restore(
                path=restore_path, 
                trainable=ray_trainer
            )
        except Exception as e:
            print(f'Error: {e}. Creating new tuner.')
            
            tuner = tune.Tuner(
                ray_trainer,
                param_space={"train_loop_config": SEARCH_SPACE},
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    num_samples=num_samples,
                    scheduler=scheduler,
                ),
            )
    else:
        tuner = tune.Tuner(
                ray_trainer,
                param_space={"train_loop_config": SEARCH_SPACE},
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    num_samples=num_samples,
                    scheduler=scheduler,
                ),
            )
        
    return tuner.fit()
    
if __name__ == '__main__':
    # Print out GPU name, if available
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
        
    # Initialize Ray
    ray.init()
    
    # Log in to wandb
    wandb.login()
    
    # Seed everything for reproducibility
    seed_value = 42
    seed_everything(seed=seed_value, workers=True)
    
    # Check if an existing tuning process exists
    experiment_path = os.path.join(CHECKPOINT_DIR, PROJECT_NAME)

    results = tune_cnn(
        config=SEARCH_SPACE,
        num_samples=NUM_SAMPLES,
        restore_path=experiment_path
    )
        
    # Get and log best result
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    # Create formatted output of best result
    best_config = best_result.config['train_loop_config']
    best_loss = best_result.metrics['val_loss']
    
    print(f'Best hyperparameters found:')
    print(f"    Learning rate: {best_config['learning_rate']}")
    print(f"    Weight decay: {best_config['weight_decay']}")
    print(f"    Kernel size: {best_config['kernel_size']}")
    print(f"    Stride: {best_config['stride']}")
    print(f"    Padding: {best_config['padding']}")
    print(f"    Batch size: {best_config['batch_size']}")
    print(f"    Scheduler factor: {best_config['scheduler_factor']}")
    print(f"    Scheduler patience: {best_config['scheduler_patience']}")
    print(f"    Scheduler threshold: {best_config['scheduler_threshold']}")
    print(f"Best loss: {best_loss:.4f}")
    
    # Save best parameters to JSON file
    base_path = os.getcwd()
    param_dir = os.path.join(base_path, 'hyper_params')
    os.makedirs(param_dir, exist_ok=True)
    
    file_name = 'best_params_convnext_large.json'
    file_path = os.path.join(param_dir, file_name)
    
    with open(file_path, 'w') as file:
        json.dump({
            'hyperparameters':{
                "learning_rate": best_config['learning_rate'],
                "weight_decay": best_config['weight_decay'],
                "kernel_size": best_config['kernel_size'],
                'stride': best_config['stride'],
                'padding': best_config['padding'],
                "batch_size": best_config['batch_size'],
                'scheduler_factor': best_config['scheduler_factor'],
                'scheduler_patience': best_config['scheduler_patience'],
                'scheduler_threshold': best_config['scheduler_threshold']
            },
            "validation_loss": best_loss
        }, file, indent=2)
        
    print(f'Wrote best hyperparameters to {file_path}')
        
