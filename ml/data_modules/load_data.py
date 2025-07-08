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

"""
load_data.py

This module defines PyTorch Lightning `DataModule` classes and custom datasets for loading
and preprocessing structured (MLP) and image-based (CNN) datasets of blood clot simulations.

Supported datasets:
- LMDBClotDataset: Loads grayscale images and log-transformed permeability from an LMDB.
- MLPClotDataset: Loads image features from a CSV file and normalizes them.
- MLPClotDataseto1: Variant of MLPClotDataset using only basic intensity features.
- MLPClotDatasetDendrogram: Uses a predefined list of features selected via hierarchical clustering.

All datamodules support train/val/test splitting and use consistent normalization.
"""

tensor_dtype = torch.bfloat16

class LMDBClotDataset(Dataset):
    """
    PyTorch Dataset for grayscale clot images stored in LMDB format.
    
    Each sample includes:
        - A transformed grayscale image tensor.
        - A permeability value.
        
    Use `set_image_dir()` to specify the LMDB path before instantiating.
    """
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
        batch_size: int=16,
        num_workers: int=70,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        seed: int=42
    ):
        """
        LightningDataModule for loading clot images from LMDB database with train/val/test splits.
        
        Parameters:
            lmdb_path (str): Path to the LMDB directory.
            batch_size (str): Batch size for DataLoaders.
            num_workers (int): Number of workers for parallel data loading.
            train_transforms, val_transforms, test_transforms (callable): Optional torchvision transforms for data augmentation.
            seed (int): Random seed for reproducible data splits.
        """
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
            
class MLPClotDataset(L.LightningDataModule):
    def __init__(self,
                 csv_path: str, 
                 batch_size: int=16, 
                 num_workers: int=70
        ):
        """
        LightningDataModule for image features stored in CSV format.
        
        Features are standardized using scikit-learn's StandardScaler, and the permeability is log-transformed. Dataset is split into 80/10/10 for train/val/test
        
        Parameters:
            csv_path (str): Path to the CSV with image features.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for parallel data loading.
        """
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        
        # TODO: Might need to make this faster/more efficient
        df = pd.read_csv(self.csv_path)
        df = df.dropna(axis=1) # Remove NaN values
        
        # Get feature columns
        start_idx = df.columns.get_loc('img_mean_intensity')
        feature_cols = df.columns[start_idx:]
        
        # Extract features and target
        self.X = df[feature_cols].values.astype('float32')
        self.y = np.log(df['k'].values.astype('float32').reshape(-1,1)) # Log-transform the permeability
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        dataset = TensorDataset(torch.tensor(self.X, dtype=tensor_dtype), torch.tensor(self.y, dtype=tensor_dtype))
        
        # Split dataset into 80% training, 10% validation, 10% testing
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


class MLPClotDataseto1(L.LightningDataModule):
    def __init__(self, 
                 csv_path: str,
                 batch_size: int=16, 
                 num_workers: int=70
    ):
        
        """
        LightningDataModule for MLP training using only the first-order statistics from 2D simulations.
        
        Loads features from a CSV file and extracts only the first-order intensity statistics from `img_mean_intensity` to `img_kurtosis_intensity` (inclusive). The permeability is log-transformed, and the features are standardized using `StandardScaler`.
        
        Dataset is split into 80/10/10 for train/val/test.
        
        Parameters:
            csv_path (str): Path to the CSV with image features.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for parallel data loading.
        """
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        
        # TODO: Might need to make this faster/more efficient
        df = pd.read_csv(self.csv_path)
        df = df.dropna(axis=1) # Remove NaN values
        
        # Get feature columns
        start_idx = df.columns.get_loc('img_mean_intensity')
        end_idx = df.columns.get_loc('img_kurtosis_intensity')

        feature_cols = df.columns[start_idx:end_idx+1]

        
        # Extract features and target
        self.X = df[feature_cols].values.astype('float32')
        self.y = np.log(df['k'].values.astype('float32').reshape(-1,1)) # Log-transform the permeability
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        dataset = TensorDataset(torch.tensor(self.X, dtype=tensor_dtype), torch.tensor(self.y, dtype=tensor_dtype))
        
        # Split dataset into 80% training, 10% validation, 10% testing
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
    
# Class to create a dataset with the image features from dendrogram
class MLPClotDatasetDendrogram(L.LightningDataModule):
    def __init__(self, 
                 csv_path: str,
                 batch_size: int=16, 
                 num_workers: int=70
    ):
        """
        LightningDataModule for MLP training using dendrogram-selected image features.
        
        Loads a CSV file and selects a subset of 12 image features identified from the dendrogram. Permeability target is log-transformed and all featured are scaled using scikit-learn's StandardScaler.
        
        Dataset is split into 80/10/10 for train/validation/testing.
        
        Parameters:
            csv_path (str): Path to the CSV with image features.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for parallel data loading.
        """
        
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
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
        
        # Split dataset into 80% training, 10% validation, 10% testing
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
