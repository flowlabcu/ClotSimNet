# Data Modules

Description of necessary files in this directory


## image_transformations

For the CNN architectures, you can perform image augmentation before passing images into the network itself. This script is called in all CNN architectures to perform image augmentation. The training transformations are more varied than the validation and testing transformations due to the desire for validation and testing transformations to be reproducible, whereas the training transformations can be stochastic.

Also note that the data types do change during the process, with bfloat16 being the final data type. This is to allow for more efficient training.

### transform_train

The transformations are the following:

* Transform image to grayscale
* Take a random 512x512 crop from the input rectangular image
* Cast it to float32
* Flip horizontally with a probability of 50%
* Flip vertically with a probability of 50%
* Randomly rotate 5 degrees
* Randomly translate the image 5% horizontally and 5% vertically, then scale the image between 0.95% and 1.05% of its original size.
* Convert the image from float32 to bfloat16 for memory efficiency
* Normalize the image with a mean of 0.5 and a standard deviation of 0.5

### transform_val

* Convert image to grayscale
* Center crop (e.g. deterministic) of 512x512 pixels
* Convert the image from float32 to bfloat16 for efficiency
* Normalize the image with a mean of 0.5 and a standard deviation of 0.5

### transform_test

* Is called in the training scripts and set equal to `transform_val`, e.g.

```python
transform_test = transform_val
```


## load_data

These are classes for loading data, both for the CNN datasets using LMDB and the MLP datasets. Both inherit Lightning's [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) class.

Classes are the following:

* `LMDBClotDataset`: Default PyTorch dataset to read in images and permeabilities from image metadata
* `LMDBClotDataModule`: Uses LMDBClotDataset but combines it with Lightning's LightningDataModule for greater efficiency and readability
* `MLPClotDataset`: Default PyTorch dataset class for the full MLP
* `MLPClotDataseto1`: Default PyTorch dataset class for the first-order MLP
* `MLPClotDatasetDendrogram`: Default PyTorch dataset class for the MLP using dendrogram features
