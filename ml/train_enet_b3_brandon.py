import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import piexif
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torch
torch.set_float32_matmul_precision('highest') # Options are medium, high, highest
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import decode_image, read_file
from torchvision.transforms import v2
import torchvision.models as models
from torchvision.io import decode_image, read_file
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from utils import export_cnn, test_report
import wandb

new_alloc = torch.cuda.memory.CUDAPluggableAllocator('./alloc.so', 'my_malloc', 'my_free')
torch.cuda.memory.change_current_allocator(new_alloc)

# TODO: Modify to allow transforms to be passed

class CNNClotDataset(Dataset):
     # Class variable to store image directory
    _image_dir = None
    
    @classmethod
    def set_image_dir(cls, image_dir):
        '''
        Class method to set the image directory once
        '''
        cls._image_dir = image_dir
        
    def __init__(self, transform=None):
        if self._image_dir is None:
            raise ValueError('Set image_dir using CNNClotDataset.set_image_dir() before creating instances')
        
        self.image_paths = [os.path.join(image_dir, file_name) 
                            for file_name in os.listdir(image_dir)
                            if file_name.endswith('jpeg')]
        
        if transform is None:
            self.transform = v2.Compose([
               v2.Grayscale(num_output_channels=1),
               v2.ToImage(),
               v2.ToDtype(torch.float32, scale=True),
               v2.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path)
        
        # Load EXIF data
        exif_dict = piexif.load(image.info['exif'])
        
        # Extract and decode permeability value
        if piexif.ExifIFD.UserComment in exif_dict['Exif']:
            metadata_str = exif_dict['Exif'][piexif.ExifIFD.UserComment].decode("utf-8")
            permeability = torch.tensor(float(metadata_str), dtype=torch.float32)
            
            # Apply transformations
            image = decode_image(read_file(image_path)).float()
            if self.transform:
                image = self.transform(image)
            
        return image, permeability
    
    
# CNN EfficientNet b0 architecture
class ENetB3(L.LightningModule):
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
        num_workers: int=4
    ):
        super().__init__()
        self.save_hyperparameters()
        
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
        self.test_preds = torch.cat(self.test_preds, dim=0).numpy()
        self.test_labels = torch.cat(self.test_labels, dim=0).numpy()

        # Store results for access after trainer.test()
        self.test_results = {'preds': self.test_preds, 'labels': self.test_labels}
    
    def setup(self, stage):
        # Load dataset
        dataset = CNNClotDataset()
        
        train_frac = 0.8
        val_frac = 0.1
        train_size = int(train_frac * len(dataset))
        val_size = int(val_frac * len(dataset))
        test_size = int(len(dataset) - train_size - val_size)
        
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])
        
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
    
    transforms = None
    data_aug = False
        
    image_dir = '/scratch/local/jogr4852_dir/min_data_1k/cnn_512_metadata'
    CNNClotDataset.set_image_dir(image_dir)
    dataset = CNNClotDataset(transform=transforms)
    
    # Hyperparams for logging
    model_type = 'ENetB3'
    dataset_size = '1k'
    
    learning_rate = 1e-3
    weight_decay = 1e-5
    kernel_size = 3
    stride = 2
    padding = 1
    bias = False
    use_weights = True
    batch_size = 128
    num_workers = 30
    max_epochs = 500
    
        
    model = ENetB3(
        use_weights=use_weights,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )

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
        'data_aug': data_aug,
        'use_weights': use_weights,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'max_epochs': max_epochs
    })
    
    wandb_logger.experiment.tags = [model_type, f'{dataset_size}_dataset', f'lr_{learning_rate}', f'bs_{batch_size}', f'{hp_tune}', f'data_aug_{data_aug}']
    
    # Add notes
    wandb_logger.experiment.notes = f'Num workers: {num_workers}; Batch size: {batch_size}; Max epochs: {max_epochs}, Tuned model: {hp_tune}'

    # Train model
    trainer = Trainer(
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
    
    model_name = 'enet_b3_base'
    model_dir = '/home/josh/clotsimnet/ml/models/'
    
    export_cnn.export(
        model=model,
        model_name=model_name,
        model_dir=model_dir,
    )
