import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from safetensors.torch import load_file
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything

import wandb

# Ensure parent directory is on path so we can import model_classes and data_modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import export_mlp, test_report
from model_classes import MLP
from data_modules import load_data

"""
train_mlp_o1_3d_scratch.py

This script traines an MLP model from scratch for predicting permeability from 3D clot data
using exclusively first-order image features. It performs k-fold cross-validation with selective layer
unfreezing, retrains the model on the full dataset, evaluates on a test set, and logs
training runs using Weights & Biases (WandB).

Main functionalities:
- Load training and test datasets from CSV files.
- Perform k-fold cross-validation (default: 5 folds), fine-tuning only the final layer.
- Retrain the model on the full dataset using tuned hyperparameters.
- Evaluate final model on an external test set.
- Generate a test report and optionally export the final model.

Expected directory layout:
- Datasets: CSV files located under `clotsimnet/data/`

Can optionally change the datasets paths to wherever you have these stored.
"""

torch.set_float32_matmul_precision('highest')  # highest precision for matmul
tensor_dtype = torch.bfloat16  # use bfloat16 where supported

def main():
    
    # Seed for reproducibility
    seed_value = 42
    seed_everything(seed=seed_value, workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters and training settings
    PROJECT_NAME = 'ClotSimNet'
    BASE_RUN_NAME = 'mlp_o1_3d_scratch_100'         # base name for WandB runs
    num_workers = 16
    max_epochs = 10
    
    # Use default hyperparameters if training from scratch
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-2
    n_folds = 5

    # Paths for pretrained safetensors and new CSV
    # Adjust these paths as needed:
    pretrained_model_dir = '/home/josh/ClotSimNet-Models/base/mlp_full/'
    pretrained_model_name = 'mlp_full_base'  # original model name (without extension)
    pretrained_path = os.path.join(pretrained_model_dir, pretrained_model_name + '.safetensors')

    # Path to small 3D CSV dataset with image features
    csv_path = '/home/josh/clotsimnet/data/mlp_data_3d_100.csv'
    test_path = '/home/josh/clotsimnet/data/mlp_data_3d.csv'
    
    # Full Dataset Dataloader
    
    data_module = load_data.MLPClotDataseto1(
        csv_path=csv_path,
        num_workers=num_workers
        )
    data_module.setup()
    
    X_np = data_module.X
    y_np = data_module.y
    # print(y_np)
    
    # Convert to tensors
    features_tensor = torch.tensor(X_np, dtype=torch.float32)
    labels_tensor = torch.tensor(y_np, dtype=torch.float32)
    
    dataset_size = features_tensor.shape[0]
    input_dim = features_tensor.shape[1]
    
    
    # Test dataset Dataloader
    test_module = load_data.MLPClotDataseto1(
        csv_path=test_path,
        num_workers=num_workers
    )
    test_module.setup()
    X_test_np = test_module.X
    y_test_np = test_module.y
    
    # Convert to tensors
    features_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    labels_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)
    
    # test_dataset_size = features_test_tensor.shape[0]
    # test_input_dim = features_test_tensor.shape[1]
    
    test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Set up K-Fold Cross-validation
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_value)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(features_tensor), start=1):
        print(f'\n=== Fold {fold} / {n_folds} ===')
        
        # Creat training/validation subsets
        train_ds = TensorDataset(features_tensor[train_idx], labels_tensor[train_idx])
        
        val_ds = TensorDataset(features_tensor[val_idx], labels_tensor[val_idx])
        
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Instantiate and load model
        model = MLP.MLP(
            input_size=input_dim
        )
        
        model = model.to(device)
        
        # Override configure_optimizers to update only unfrozen parameters
        import types
        def _configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                ),
                'monitor':'val_loss'
            }
            return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor':'val_loss'}
        
        model.configure_optimizers = types.MethodType(_configure_optimizers, model)
        
        # Set up logger and trainer
        # run_name = f'{BASE_RUN_NAME}_fold_{fold}'
        # wandb_logger = WandbLogger(project=PROJECT_NAME, name=run_name)
        # wandb_logger.log_hyperparams({
        #     'fold':fold,
        #     'learning_rate':learning_rate,
        #     'weight_decay':weight_decay,
        #     'batch_size':batch_size,
        #     'max_epochs':max_epochs
        # })
        
        trainer = Trainer(
            precision='bf16-mixed',
            max_epochs=max_epochs,
            accelerator='auto',
            devices='auto',
            strategy='auto',
            # logger=wandb_logger,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=True)],
            log_every_n_steps=1
        )
        
        # Fit and validate model
        trainer.fit(model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader
        )
        
        # Record the best validation loss from this fold
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            fold_metrics.append(val_loss.item())
            print(f'Fold {fold} validation loss: {val_loss.item():.6f}')
        else:
            fold_metrics.append(None)
            print(f'Fold {fold}: val_loss not recorded')
            
        # Finish WandB run for this fold
        # wandb_logger.experiment.finish()
        
    # Report CV results
    valid_losses = [v for v in fold_metrics if v is not None]
    if valid_losses:
        mean_loss = np.mean(valid_losses)
        median_loss = np.median(valid_losses)
        std_loss = np.std(valid_losses)
        
        print(f"\nCross-Validation Results (val_loss) mean ± STD: {mean_loss:.6f} ± {std_loss:.6f}")
        
        print(f"\nCross-Validation Results (val_loss) median ± STD: {median_loss:.6f} ± {std_loss:.6f}")
        
    else:
        print('\nNo validation losses recorded during CV')
        
    # Retrain on full dataset
    print('\n=== Retraining on full dataset ===')
    
    # Instantiate fresh MLP
    final_model = MLP.MLP(input_size=input_dim)
    final_model = final_model.to(device)
    
    # Override configure_optimizers for final model
    import types
    def _configure_optimizers_full(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
            ),
            'monitor':'train_loss' # Monitor training loss since no validation on final re-train
        }
        return {'optimizer':optimizer, 'lr_scheduler':scheduler, 'monitor':'train_loss'}

    final_model.configure_optimizers = types.MethodType(_configure_optimizers_full, final_model)
    
    # Create DataLoader for full dataset
    full_dataset = TensorDataset(features_tensor, labels_tensor)
    full_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Set up WandB logger and Trainer for final retrain
    wandb_logger_full = WandbLogger(
        project=PROJECT_NAME,
        name='final_retrain_50'
    )
    wandb_logger_full.log_hyperparams({
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'max_epochs': max_epochs
    })
    
    trainer_full = Trainer(
            precision='bf16-mixed',
            max_epochs=max_epochs,
            accelerator='auto',
            devices='auto',
            strategy='auto',
            logger=wandb_logger_full,
            callbacks=[EarlyStopping(monitor='train_loss', patience=3, verbose=True)],
            log_every_n_steps=1
    )
    
    # Train on full dataset
    trainer_full.fit(final_model, 
                     train_dataloaders=full_loader
    )

    # Test model on 10 example dataset
    trainer_full.test(
        final_model,
        dataloaders=test_loader,
        verbose=True
    )
    
    preds = final_model.test_results['preds']
    labels = final_model.test_results['labels']
    
    # Create the nicely formatted test report
    test_report.create_report(preds=preds, labels=labels)
    
    wandb_logger_full.experiment.finish()
    
    # Save final model
    # new_model_name = BASE_RUN_NAME
    # export_mlp.export(
    #     model=final_model,
    #     model_name=new_model_name,
    #     model_dir=pretrained_model_dir,
    #     input_size=input_dim
    # )


if __name__ == '__main__':
    main()