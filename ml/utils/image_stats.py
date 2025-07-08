# Code source: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/19?u=kuzand
# Code modified for single-channel/grayscale input images
import os, torch, torchvision
from torchvision.io import decode_image, read_file, image
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

"""
image_stats.py

This is a script to calculate the mean and standard deviation pixel intensities of an image dataset to be used in the v2.Normalize() torchvision transform calls. It is recommended to try these numbers in the CNN data augmentation pipeline to see if it improves model performance over the default 0.5, 0.5 values.

Testing showed that the default values of 0.5, 0.5 worked better in all CNN models than what was calculated from this script. Dataset size in 2D was ~5,000, although the dataset mean and standard deviation as calculated in this script may prove to help models perform better as dataset size increases.
"""

class MyDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        transform=None
    ):  
        """
        PyTorch Dataset class to loop through image folder and optionally transform it.
        
        Parameters:
            image_dir (str): Path to directory containing CNN images
            transform: Optional data augmentation transforms. Defaults to None.
        """
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

# Create Dataset
dataset = MyDataset(
    image_dir='/home/josh/clotsimnet/data/cnn_data_crop',
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

# Create DataLoader from Dataset
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=0,
    shuffle=False
)

# Calculate dataset mean and standard deviation in batches
mean = 0.
nb_samples = 0.
print('Calculating mean:')
for data in tqdm(loader):
    batch_samples = data.size(0) # Pull out batch size
    data = data.view(batch_samples, data.size(1), -1) # Convert data to a vector
    mean += data.mean(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples

temp = 0.
nb_samples = 0.
print('Calculating standard deviation')
for data in tqdm(loader):
    batch_samples = data.size(0) # Extract the batch size
    elementNum = data.size(0) * data.size(2) * data.size(3) # Multiply batch_size * height * width
    data = data.view(1, -1)
    temp += ((data - mean)**2).sum(1)/(elementNum*batch_samples)
    nb_samples += batch_samples

std = torch.sqrt(temp/nb_samples)
print(f'Mean: {mean.item()}')
print(f'Standard deviation: {std.item()}')