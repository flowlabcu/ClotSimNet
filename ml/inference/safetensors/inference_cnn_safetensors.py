import os, sys, torch, torchvision
from safetensors.torch import load_file 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..', '..')))          
from data_modules import image_transformations   
from model_classes import EfficientNets, ResNets, ConvNeXt           
from PIL import Image
import pandas as pd
import numpy as np
import piexif
from tqdm import tqdm

def run_inference_cnn_base(model_name: str, state_dict: str, images_dir: str, save_file: str):
    if model_name == 'ENet-B0':
        model = EfficientNets.ENetB0()
    elif model_name == 'ENet-B3':
        model = EfficientNets.ENetB3()
    elif model_name == 'ENet-B7':
        model = EfficientNets.ENetB7()
    elif model_name == 'ResNet-18':
        model = ResNets.ResNet18()
    elif model_name == 'ResNet-50':
        model = ResNets.ResNet50()
    elif model_name == 'ResNet-152':
        model = ResNets.ResNet152()
    elif model_name == 'ConvNeXt-Tiny':
        model = ConvNeXt.ConvNeXtTiny()
    else:
        print('Model not found')
        print('Models available:')
        print('ENet-B0: \t\t EfficientNet-B0')
        print('ENet-B3: \t\t EfficientNet-B3')
        print('ENet-B7: \t\t EfficientNet-B7')
        print('ResNet-18: \t\t ResNet-18')
        print('ResNet-50: \t\t ResNet-50')
        print('ResNet-152: \t\t ResNet-152')
        print('ConvNeXt-Tiny: \t\t ConvNeXt Tiny')
        
    state_dict = load_file(
    state_dict,
    device="cpu" # keep on CPU while loading
    )
    
    print(f'Running inference on model {model_name}')
    
    model.load_state_dict(state_dict, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device).eval() # Send model to GPU or CPU and put in evaluation mode
    
    val_tf = image_transformations.transform_val()    # Load in validation transforms for consistent evaluations

    results = []
    for fn in tqdm(sorted(os.listdir(images_dir)), desc="Processing images", unit=" images"):
        fullpath = os.path.join(images_dir, fn)
        
        # Extract permeability from metadata in each image using EXIF (UserComment field)
    
        pil = Image.open(fullpath)
        label_k = None
        exif_bytes = pil.info.get('exif')
        if exif_bytes:
            exif_dict = piexif.load(exif_bytes)
            user_comment = exif_dict['Exif'].get(piexif.ExifIFD.UserComment)
            if user_comment:
                label_k = float(user_comment.decode('utf-8'))

        # Load the image as a tensor, convert to grayscale
        img_tensor = torchvision.io.read_image(
        fullpath, mode=torchvision.io.image.ImageReadMode.GRAY
        ).float()

        # Pass image through the validation image transformation pipeline
        x = val_tf(img_tensor)           # now shape [1,512,512], dtype=bfloat16
        x = x.to(torch.float32)
        x = x.unsqueeze(0).to(device)    # Send image to the GPU or CPU

        # --- inference ---
        with torch.no_grad():
            logk = model(x)              # network trained on out log(k)
            pred_k = float(logk.exp().cpu())

        # Append the results to the results dictionary
        results.append({
            'filename':  os.path.basename(fn),
            'permeability': label_k,
            'prediction': pred_k,
        })
        
    # Save predictions to CSV
    pd.options.display.float_format = '{:.6e}'.format
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False, float_format='%.6e')
    print(f'Predictions saved to {save_file}')
        
       

images_dir = '/home/josh/clotsimnet/data/cnn_data_test/cnn_data_crop'

run_inference_cnn_base(model_name='ENet-B0',
                  state_dict='/home/josh/ClotSimNet-Models/base/enet_b0/enet_b0_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='enet_b0_base_preds.csv')

run_inference_cnn_base(model_name='ENet-B3',
                  state_dict='/home/josh/ClotSimNet-Models/base/enet_b3/enet_b3_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='enet_b3_base_preds.csv')

run_inference_cnn_base(model_name='ENet-B7',
                  state_dict='/home/josh/ClotSimNet-Models/base/enet_b7/enet_b7_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='enet_b7_base_preds.csv')

run_inference_cnn_base(model_name='ResNet-18',
                  state_dict='/home/josh/ClotSimNet-Models/base/resnet_18/resnet_18_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='resnet_18_base_preds.csv')

run_inference_cnn_base(model_name='ResNet-50',
                  state_dict='/home/josh/ClotSimNet-Models/base/resnet_50/resnet_50_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='resnet_50_base_preds.csv')

run_inference_cnn_base(model_name='ResNet-152',
                  state_dict='/home/josh/ClotSimNet-Models/base/resnet_152/resnet_152_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='resnet_152_base_preds.csv')

run_inference_cnn_base(model_name='ConvNeXt-Tiny',
                  state_dict='/home/josh/ClotSimNet-Models/base/convnext_tiny/convnext_tiny_base.safetensors', 
                  images_dir=images_dir, 
                  save_file='convnext_tiny_base_preds.csv')
