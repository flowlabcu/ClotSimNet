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
from utils import export_cnn, test_report
import wandb
import lmdb
import msgpack
import sys
import torchvision

torch.set_float32_matmul_precision('highest') # Options are medium, high, highest

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16

# For training on AMD GPUs, as bfloat16 isn't supported
# tensor_dtype = torch.float16


class LMDBClotDataset(Dataset):
    # Class variable to store image directory
    _lmdb_path = None
    
    @classmethod
    def set_image_dir(cls, lmdb_path):
        '''
        Class method to set the LMDB directory once
        '''
        cls._lmdb_path = lmdb_path
        
    def __init__(self, transform=None, indices=None):
        if self._lmdb_path is None:
            raise ValueError('Set lmdb_path using CNNClotDataset.set_lmdb_path() before creating instances')
        
        self.transform = transform
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
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
        batch_size: int=4,
        num_workers: int=18,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        seed: int=42
    ):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.batch_size = batch_size
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
    
# CNN EfficientNet b0 architecture
class ENetB0(L.LightningModule):
    def __init__(
        self, 
        learning_rate: float=1e-3, 
        weight_decay: float=1e-5, 
        kernel_size: int=3, 
        stride: int=2, 
        padding: int=1,
        bias: bool=False, 
        use_weights: bool=True,
        batch_size: int=256, 
        num_workers: int=4,
        seed: int=42   
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Transforms flag to perform image augmentations
        self.use_transforms = use_transforms
        
        # Save predictions for later
        self.test_preds = []
        self.test_labels = []
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.use_weights = use_weights
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
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
            padding=padding, 
            bias=bias
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
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.012840669372724956, patience=40, threshold=0.006612070221028295)
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'}
    
if __name__ == '__main__':
    # Seed everything for reproducibility
    seed_value = 42
    seed_everything(seed=seed_value, workers=True)
    
    # Define transforms
    transform_train = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        
        # Use float32 for geometric transforms
        v2.ToDtype(torch.float32, scale=True),
        
        # Augmentations for training only
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=5),
        v2.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        
        # Convert to bfloat16 for rest of pipeline
        v2.ConvertImageDtype(dtype=tensor_dtype),
        # v2.ColorJitter(
        #     brightness=0.1,
        #     contrast=0.1
        #     ),
        v2.GaussianNoise(mean=0.0, sigma=0.01),
        
        # Final normalization (scale to [-1, 1] range)
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    # Validation/test transforms (only preprocessing, no data augmentation)
    transform_val = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        v2.ToDtype(tensor_dtype, scale=True),
        
        # Final normalization (scale to [-1, 1] range)
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_test = transform_val
    
    lmdb_path = '/scratch/local/jogr4852_dir/data_250.lmdb'
    LMDBClotDataset.set_image_dir(lmdb_path)
    
    # Hyperparams for logging
    model_type = 'ENetB0'
    dataset_size = '250'
    
    learning_rate = 0.054480726081375434
    weight_decay = 0.008241502276709554
    kernel_size = 7
    stride = 3
    padding = 12
    bias = False
    use_weights = True
    batch_size = 16
    num_workers = 18
    max_epochs = 500
    use_transforms = False
        
    model = ENetB0(
        use_weights=use_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    
    data_module = LMDBClotDataModule(
        lmdb_path='/scratch/local/jogr4852_dir/data_250.lmdb',
        batch_size=batch_size,
        num_workers=num_workers,
        train_transforms=transform_train,
        val_transforms=transform_val,
        test_transforms=transform_test
    )

    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
        
    # Start logger    
    hp_tune = 'tuned' # base vs. tuned model
    
    wandb_logger = WandbLogger(
        project='ENet B0 Debugging', 
        name='ENet B0 No Aug'
    )

    wandb_logger.log_hyperparams({
        'model': model_type,
        'dataset_size': dataset_size,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'bias': bias,
        'data_aug': use_transforms,
        'use_weights': use_weights,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'max_epochs': max_epochs
    })
    
    wandb_logger.experiment.tags = [model_type, f'{dataset_size}_dataset', f'lr_{learning_rate}', f'bs_{batch_size}', f'{hp_tune}', f'data_aug_{use_transforms}']
    
    # Add notes
    wandb_logger.experiment.notes = f'Num workers: {num_workers}; Batch size: {batch_size}; Max epochs: {max_epochs}, Tuned model: {hp_tune}'

    # Train model
    trainer = Trainer(
        precision="bf16-mixed",
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=wandb_logger, # Comment out to remove logging
        log_every_n_steps=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=100, verbose=True)]
        )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, data_module)
    
    preds = model.test_results['preds']
    labels = model.test_results['labels']
    
    test_report.create_report(preds=preds, labels=labels)
    
    # Save and export model
#    model_name = 'enet_b0_tuned_aug'
#    model_dir = '/home/josh/clotsimnet/ml/models/enet_b0'
    
#    export_cnn.export(
#        model=model,
#        model_name=model_name,
#        model_dir=model_dir
#    )
        
