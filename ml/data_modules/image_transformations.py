import torch, torchvision
from torchvision.io import decode_image, read_file, image
from torchvision.transforms import v2

tensor_dtype = torch.bfloat16 # Can change to torch.bfloat16

# For training on AMD GPUs, as bfloat16 isn't supported
# tensor_dtype = torch.float16

def transform_train():
    """
    Creates and returns a torchvision image trasnformation pipeline for training grayscale images.
    
    Parameters:
        None
        
    Returns:
        torchvision.transforms.v2.Compose: Composed transformation object for training images
    """
    return v2.Compose([
        v2.Grayscale(num_output_channels=1), # Convert image to grayscale
        v2.ToImage(),
        v2.RandomCrop((512, 512)), # Random crop of 512x512 pixels to fit within expected input dimension of CNNs
        
        # Use float32 for geometric transforms
        v2.ToDtype(torch.float32, scale=True),
        
        # Augmentations for training only
        v2.RandomHorizontalFlip(p=0.5), # 50% probability
        v2.RandomVerticalFlip(p=0.5),
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
        # v2.GaussianNoise(mean=0.0, sigma=0.01),
        
        # Final normalization (scale to [-1, 1] range)
        # Original values: mean=[0.5], std=[0.5]
        v2.Normalize(mean=[0.5], std=[0.5]) # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/6?u=kharshit
    ])

def transform_val():
    """
    Creates and returns a torchvision image trasnformation pipeline for validation dataset on grayscale images.
    
    Unlike transform_train, only deterministic transformations are done here to ensure consistent model evaluations.
    
    Parameters:
        None
        
    Returns:
        torchvision.transforms.v2.Compose: Composed transformation object for training images
    """
    return v2.Compose([
        v2.Grayscale(num_output_channels=1), # Convert to grayscale
        v2.ToImage(),
        v2.CenterCrop((512, 512)), # Center crop 512x512 pixels
        v2.ToDtype(tensor_dtype, scale=True),
        
        # Final normalization (scale to [-1, 1] range)
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
