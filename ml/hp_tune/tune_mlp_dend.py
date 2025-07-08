import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
import wandb
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

# Dynamically construct the path based on the user's home directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from utils import export_mlp, test_report
from model_classes import MLP
from data_modules import image_transformations, load_data

"""
tune_mlp_dend.py

This script tunes the dendrogram MLP on engineered tabular features for predicting permeability from CFD-derived clot simulations. It uses Ray Tune with the ASHA scheduler to search over hyperparameters and leverages PyTorch Lightning and Weights & Biases (W&B) for scalable training and experiment tracking.

Key Features:
-------------
- Reads a CSV of precomputed image/dendrogram features (intensity stats, GLCM, Laws textures) and log-transformed permeability labels.
- Normalizes features using StandardScaler.
- Dynamically constructs an MLP with configurable hidden size and number of layers.
- Hyperparameter search over:
    • hidden_size
    • num_layers
    • learning_rate
    • weight_decay
    • batch_size
    • scheduler_factor, scheduler_patience, scheduler_threshold
- Tracks regression metrics (MAE, RMSE, MSE, R²) via TorchMetrics.
- Saves best hyperparameters and validation loss to 
  `hyper_params/best_params_mlp_dend.json`.

Required:
---------
- A CSV file with feature columns matching the hardcoded `feature_list` in `MLPClotDatasetDendrogram.setup` and a `k` column for permeability.
- W&B API key and login (or set `WANDB_API_KEY` env var). Can uncomment if using something else, or if not used at all. Note that you will also need to uncomment the `logger` flags as well.
- (Optional) GPUs for mixed-precision (bfloat16) training. Can also use float16 if using AMD GPUs (currently, only NVIDIA supports bfloat16)

Usage:
------
Run the tuning job:
    python tune_mlp_o1.py

If interrupted, it will restore from:
    CHECKPOINT_DIR/PROJECT_NAME

Output:
-------
- Prints best hyperparameters and loss.
- Persists best hyperparameters as JSON.
- Logs all runs under the `ray_tune_runs` group in W&B.

Notes:
------
- Feature extraction is currently fixed via a hardcoded list, modify as needed.
- Adjust `NUM_EPOCHS`, `NUM_SAMPLES`, `DEFAULT_CONFIG`, `SEARCH_SPACE`, 
  `csv_path`, and `CHECKPOINT_DIR` to customize your experiments.
"""

torch.set_float32_matmul_precision('highest') # Options are medium, high, highest

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16 or torch.float32


### ----- Define Search Space for Tuning ---- ###

# Configuration settings
# Configuration settings
DEFAULT_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-2,
    'batch_size': 32,
    'scheduler_factor': 0.1,
    'scheduler_patience': 10,
    'scheduler_threshold': 1e-4,
}

SEARCH_SPACE = {
        'hidden_size': tune.qlograndint(128, 4096, 4), 
        'num_layers': tune.randint(1, 5),
        'learning_rate': tune.loguniform(1e-5, 1e-3), 
        'weight_decay': tune.loguniform(1e-4, 1e-2),
        'batch_size': tune.choice([32, 64, 128, 256, 512]),
        'scheduler_factor': tune.loguniform(0.1, 0.2, 0.5),
        'scheduler_patience': tune.randint(5, 16),
        'scheduler_threshold': tune.loguniform(1e-4, 1e-2)
}

# Training settings
NUM_EPOCHS = 70 # Number of training epochs
NUM_SAMPLES = 500 # Number of samples from parameter space

CHECKPOINT_DIR = '/scratch/alpine/jogr4852/ray_checkpoints/'
# CHECKPOINT_DIR = '/home/josh/ray_checkpoints'

PROJECT_NAME = 'tune_mlp_dend'

SCALING_CONFIG = ScalingConfig(
    num_workers=1, 
    use_gpu=True, 
    resources_per_worker={"GPU": 1, "CPU": 64}
)

CHECKPOINT_CONFIG = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute='val_loss',
    checkpoint_score_order='min',
    checkpoint_frequency=0
)

class MLPClotDatasetDendrogram(L.LightningDataModule):
    def __init__(self, csv_path: str, config, num_workers: int):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = config['batch_size']
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        
        # TODO: Might need to make this faster/more efficient
        df = pd.read_csv(self.csv_path)
        df = df.dropna(axis=1) # Remove NaN values
        
        # Get feature columns
        feature_list = [
            'img_median_intensity',
            'img_mean_intensity',
            'glcm_homogeneity_dist_10_angle_0',
            'laws_L5R5_div_R5L5_mean',
            'img_variance_intensity',
            'img_std_intensity',
            'glcm_contrast_dist_5_angle_0',
            'laws_E5E5_mean',
            'img_kurtosis_intensity',
            'img_max_intensity',
            'glcm_correlation_dist_1_angle_0',
            'glcm_correlation_dist_50_angle_2_3562'
        ]

        feature_cols = df[feature_list].columns
        
        # Extract features and target
        self.X = df[feature_cols].values.astype('float32')
        self.y = np.log(df['k'].values.astype('float32').reshape(-1,1)) # Log-transform the permeability
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        dataset = TensorDataset(torch.tensor(self.X, dtype=tensor_dtype), torch.tensor(self.y, dtype=tensor_dtype))
        
        train_frac = 0.8
        val_frac = 0.1
        train_size = int(train_frac * len(dataset))
        val_size = int(val_frac * len(dataset))
        test_size = int(len(dataset) - train_size - val_size)
        
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])
        
    # NOTE: pin_memory=True calls in the next three defs should be monitored for performance. Can easily set to false.
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

# Model architecture
class MLP(L.LightningModule):
    def __init__(
        self, 
        input_size: int,
        config,
        num_workers: int=4,
        seed: int=42
    ):
        super().__init__()
        self.save_hyperparameters() # Save model hyperparameters for checkpointing
        
        # Set attributes to pull from Ray Tune configuration file
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.scheduler_factor = config['scheduler_factor']
        self.scheduler_patience = config['scheduler_patience']
        self.scheduler_threshold = config['scheduler_threshold']
        
        # Non-tuned attributes  
        self.num_workers = num_workers
        self.input_size = input_size
        self.seed = seed
        
        # Save predictions for later
        self.test_preds = []
        self.test_labels = []
        
        # Define model layers
        layers = []
        
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        
        # Set number of hidden layers dynamically via class definition
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.BatchNorm1d(num_features=self.hidden_size))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(self.hidden_size, 1)) # Output layer for regression and permeability
        
        self.model = nn.Sequential(*layers) # Build model
        
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
        return self.model(x)

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


        # Store results for access after trainer.test()
        self.test_results = {'preds': self.test_preds, 'labels': self.test_labels}
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, threshold=self.scheduler_threshold)
    
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'}
    
    # NOTE: pin_memory=True calls in the next three defs should be monitored for performance. Can easily set to false.
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
    
    
def train_func(config):
    # Create a name for each trial based on hyperparameters being trialed
    # name = f"hs_{config['hidden_size']}_num_layers_{config['num_layers']}_lr_{config['lr']:.4f}_weight_decay_{config['weight_decay']}_bs_{config['batch_size']}"
    
    # Create new WandBLogger for training run
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        # name=name,
        reinit=True
    )
    
    # Log hyperparameters for this run
    wandb_logger.log_hyperparams({
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'batch_size': config['batch_size'],
        'scheduler_factor': config['scheduler_factor'],
        'scheduler_patience': config['scheduler_patience'],
        'scheduler_threshold': config['scheduler_threshold']
    })
    
    # Seed everything for reproducibility
    seed_value = 42
    seed_everything(seed=seed_value, workers=True)
    
    # Load dataset
    # csv_path = '/home/josh/clotsimnet/data/mlp_data_5k.csv'
    csv_path = '/scratch/local/jogr4852_dir/mlp_data_5k.csv'
    # batch_size = 5040
    num_workers = 64
    
    data_module = MLPClotDatasetDendrogram(
        csv_path=csv_path, 
        config=config,  
        num_workers=num_workers
    )
    data_module.setup()
    
    # Create model
    model = MLP(
        input_size=data_module.X.shape[1],
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
        log_every_n_steps=1
        )
    
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
def tune_mlp(config, num_samples=NUM_SAMPLES, restore_path=None):
    scheduler = ASHAScheduler(max_t=NUM_EPOCHS, grace_period=1, reduction_factor=2)
    
    # Define trial name generator for each individual run
    # def trial_name_generator(trial):
    #     config = trial.config['train_loop_config']
    #     name = f"hs_{config['hidden_size']}_num_layers_{config['num_layers']}_lr_{config['lr']:.4f}_weight_decay_{config['weight_decay']}_bs_{config['batch_size']}"
    #     return name 
    
    # Create WandbLoggerCallback
    wandb_callback = WandbLoggerCallback(
        project=PROJECT_NAME,
        api_key=None,
        log_config=True,
        resume='allow',
        # name_generator=trial_name_generator,
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
    
    results = tune_mlp(
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
    print(f"    Hidden size: {best_config['hidden_size']}")
    print(f"    Number of layers: {best_config['num_layers']}")
    print(f"    Learning rate: {best_config['learning_rate']}")
    print(f"    Weight decay: {best_config['weight_decay']}")
    print(f"    Batch size: {best_config['batch_size']}")
    print(f"    Scheduler factor: {best_config['scheduler_factor']}")
    print(f"    Scheduler patience: {best_config['scheduler_patience']}")
    print(f"    Scheduler threshold: {best_config['scheduler_threshold']}")
    print(f"Best loss: {best_loss:.4f}")
    
    # Save best parameters to JSON file
    base_path = os.getcwd()
    param_dir = os.path.join(base_path, 'hyper_params')
    os.makedirs(param_dir, exist_ok=True)
    
    file_name = 'best_params_mlp_dend.json'
    file_path = os.path.join(param_dir, file_name)
    
    with open(file_path, 'w') as file:
        json.dump({
            'hyperparameters':{
                "hidden_size": best_config['hidden_size'],
                "num_layers": best_config['num_layers'],
                "learning_rate": best_config['learning_rate'],
                "weight_decay": best_config['weight_decay'],
                "batch_size": best_config['batch_size'],
                'scheduler_factor': best_config['scheduler_factor'],
                'scheduler_patience': best_config['scheduler_patience'],
                'scheduler_threshold': best_config['scheduler_threshold']
            },
            "validation_loss": best_loss
        }, file, indent=2)
        
    print(f'Wrote best hyperparameters to {file_path}')
