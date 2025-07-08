# ML Folder Overview
Josh Gregory

Last edited: June 27, 2025

Description: Overview of the `ml` folder and its subdirectories

## Directory Structure

`data_modules`
- Contains scripts for data augmentation and loading data.

`hp_tune`
- Scripts for hyperarameter tuning all models.
- Contains both Python files as well as bash scripts to run in a SLURM-managed cluster environment.
- Contained `hyper_params` folder where each hyperparameter JSON file is stored after model has finished tuning.

`inference`
- Scripts for performing inference on all of the trained models.
    - `onnx` sub-directory is for performing inference on models stored in `.onnx` format 
    - `pytorch_default` sub-directory is for performing inferene on PyTorch's default `state_dict()` format
    - `safetensors` sub-directory is for performing inference on models exported in `.safetensors` format
    `torchscript` sub-directory is for performing inference on models in TorchScript format.
- `prediction_plots`: Stores all plots that show model's predictions vs. labels after running inference on each model.
- `plot_preds.py`: Plots all of the prediction plots stored in `prediction_plots` sub-directory.
- `pred_stats.py`: Calculates statistics for each model on the evaluation dataset. Runs a paired $t$-test on differences between model's predictions and labels (returning $t$-statistic and $p$-values), calculates Mean Absolute Error (MAE) between model's predictions and labeled values, and calculates $R^2$ scores for each model. Does this for all base and tuned models, as well as both base and tuned MLP models with varying amounts of data augmentation.
- **Note**: All inference was performed on the models exported in `.safetensors` format, although any format can be used; scripts to run inference on any file format are included in each sub-directory.

`lmdb_datastet`
- Contains `write_lmdb.py`, which takes a directory of images with permeability embedded in each image's metadata and converts it to LMDB format.

`model_classes`
- Contains class definitions for all of the models used (ConvNext, EfficientNet, ResNet, and all three MLPs).

`old`
- Directory containing old scripts. **Can be ignored**.

`testing`
- Directory containing very rough testing/development scripts. **Can be ignored**.

`train_base`
- Contains all of the scripts to train the base models, both Python files and bash scripts to run in a SLURM-managed cluster environment.
- `mlp_3d`: Directory containing scripts to train the MLPs on 3D simulation data. Refer to the README.md file within this directory for more details.

`train_tuned`
- Contains all of the scripts to train the hyperparameter tuned models.
- Includes both Python scripts and bash scripts to run in a SLURM-managed cluster environment.

`utils`

Contains various utility scripts

- `old`: Contains old scripts. **Can be ignored**
- `co2_estimate.py`: Script to estimate the CO2 released by a model during training/hyperparameter tuning. Requires the power curve CSV file to be downloaded from Weights & Biases (WandB) logging.
- `export_cnn.py`: Script to export CNN models in various model formats after training for later use and inference.
- `export_mlp.py`: Script to export MLP models in various model formats after training for later use and inference.
- `image_stats.py`: Script to calculate the mean and standard deviation pixel intensities from an image dataset. Recommended to use in the torchvision `v2.Normalize()` call. Default is 0.5, 0.5, which was found to work better in our datasets, although the values from this script may need to be used as dataset size increases.
- `prepare_mlp.py`: Script to take all `data.csv` files and combine them into a master CSV file for MLP training.
- `test_report.py`: Creates a nice report comparing model predictions against labeled values. Called by models at the end of training for quick performance assessments.

`xai`

Short for "Explainable AI". Contains all of the scripts needed to perform model interpretability.

- `grad_cam_plots`: Contains images of all GradCam interpretability maps.
- `old`: Contains old scripts. **Can be ignored**.

All other files are for interpretability of their respected models. Scripts that end in `_base` are for interpretability on the base models, `_tuned` are for interpretability on the models that have been hyperparameter tuned. I know it's disgusting that each model and variant has its own script. I tried combining them but ran out of time :( Feel free to do that though, each of the scripts are pretty similar.
