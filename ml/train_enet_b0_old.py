import os
import numpy as np
import pandas as pd
import piexif
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torch
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

# Define transforms
transform_train = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    
    # Use float32 for geometric transforms
    v2.ToDtype(torch.float32, scale=True),
    
    # Augmentations for training only
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    
    # Convert to bfloat16 for rest of pipeline
    v2.ConvertImageDtype(dtype=tensor_dtype),
    v2.ColorJitter(
        brightness=0.1,
        contrast=0.1
        ),
    v2.GaussianNoise(mean=0.0, sigma=0.03),
    
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


class LMDBClotDataset(Dataset):
    # Class variable to store image directory
    _image_dir = None
    
    @classmethod
    def set_image_dir(cls, lmdb_path):
        '''
        Class method to set the LMDB directory once
        '''
        cls._lmdb_path = lmdb_path
        
    def __init__(self, transform=False):
        if self._lmdb_path is None:
            raise ValueError('Set lmdb_path using CNNClotDataset.set_lmdb_path() before creating instances')
        
        self.transform = transform
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        with self.env.begin(write=False) as txn:
            self.keys = list(txn.cursor().iternext(values=False)) # Get all keys/image names
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            data = msgpack.unpackb(txn.get(f'{idx}'.encode()), raw=False) # Load pickled data
        
        # Extract image bytes
        image_bytes = data['image']
        
        # Convert to tensor
        image_tensor = torchvision.io.decode_image(torch.tensor(bytearray(image_bytes), dtype=torch.uint8),
                                                   mode=torchvision.io.image.ImageReadMode.GRAY).float()

        # Apply transformations
        if self.transform:
            image = self.transform(image_tensor)
        
        permeability = np.log(data['permeability']) # Log-transform the permeability
        permeability = torch.tensor(float(permeability), dtype=tensor_dtype)
            
        return image, permeability
    
    
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
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 1) # Single output for regression
        
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # Log MSE loss for monitoring
        
        # Log metrics
        batch_values = self.train_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True)
        return loss

    # Reset metrics
    def on_validation_epoch_start(self):
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        
        # Update validation metrics
        batch_values = self.valid_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True)
        
        return loss
    
    # Reset metrics
    def on_test_epoch_start(self):
        self.test_metrics.reset()
        
    def test_step(self, batch, batch_idx):
        x, y = batch # Get features and labels
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        
        # Update validation metrics
        batch_values = self.test_metrics(y_pred, y)
        self.log_dict(batch_values, on_step=True, on_epoch=True)
        
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
    
    def setup(self, stage):
        # Training validation split, remainder is for the test set
        train_frac = 0.8
        val_frac = 0.1
        
        # Get the trasnforms flag from the parent script
        use_transforms = getattr(self, 'use_transforms', False)
        
        if use_transforms:
            # Load datasets each with different image transforms
            train_dataset = LMDBClotDataset(transform=transform_train)
            val_dataset = LMDBClotDataset(transform=transform_val)
            test_dataset = LMDBClotDataset(transform=transform_val)
        
        else:
            # Create datasets with minimal transforms
            min_transform = v2.Compose([
               v2.Grayscale(num_output_channels=1),
               v2.ToImage(),
               v2.ToDtype(tensor_dtype, scale=True),
               v2.Normalize(mean=[0.5], std=[0.5])
            ])
            
            train_dataset = LMDBClotDataset(transform=min_transform)
            val_dataset = LMDBClotDataset(transform=min_transform)
            test_dataset = LMDBClotDataset(transform=min_transform)
            
        # Use a full dataset to get correct index splits
        full_dataset = LMDBClotDataset(transform=None)
        total_size = len(full_dataset)
        
        train_size = int(train_frac * len(full_dataset))
        val_size = int(val_frac * len(full_dataset))
        test_size = int(len(full_dataset) - train_size - val_size)
        
        # Get indices for the dataset splits
        indices = list(range(total_size))
        
        # Generate fixed seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        train_indices, val_test_indices = random_split(indices, [train_size, val_size + test_size], generator=generator)
        val_indices, test_indices = random_split(val_test_indices, [val_size, test_size], generator=generator)
        
        # Create dataset subsets
        self.train_set = Subset(train_dataset, train_indices.indices)
        self.val_set = Subset(val_dataset, val_indices.indices)
        self.test_set = Subset(test_dataset, test_indices.indices)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    # NOTE: pin_memory=True calls in the next three defs should be monitored for performance. Can easily set to false.
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
    
if __name__ == '__main__':
    # Seed everything for reproducibility
    seed_value = 42
    seed_everything(seed=seed_value, workers=True)
    
    lmdb_path = '/home/josh/clotsimnet/data_5k.lmdb'
    LMDBClotDataset.set_image_dir(lmdb_path)
    
    # Hyperparams for logging
    model_type = 'ENetB0_log'
    dataset_size = '5k'
    
    learning_rate = 1e-3
    weight_decay = 1e-5
    kernel_size = 3
    stride = 2
    padding = 1
    bias = False
    use_weights = True
    batch_size = 12
    num_workers = 18
    max_epochs = 500
    use_transforms = False
        
    model = ENetB0(
        use_weights=use_weights,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        seed=seed_value
    )
    model.use_transforms = use_transforms

    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")
        
    # Start logger    
    hp_tune = 'base' # base vs. tuned model
    
    wandb_logger = WandbLogger(
        project='ClotSimNet', 
        name=f'{model_type}_{dataset_size}_bs{batch_size}_{hp_tune}' 
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
        callbacks=[EarlyStopping(monitor='val_loss', patience=40, verbose=True)]
        )

    trainer.fit(model)
    trainer.test(model)
    
    preds = model.test_results['preds']
    labels = model.test_results['labels']
    
    test_report.create_report(preds=preds, labels=labels)
    
    # Save and export model
    model_name = 'enet_b0_base_5k_log'
    model_dir = '/home/josh/clotsimnet/ml/models/enet_b0'
    
    export_cnn.export(
        model=model,
        model_name=model_name,
        model_dir=model_dir
    )
        