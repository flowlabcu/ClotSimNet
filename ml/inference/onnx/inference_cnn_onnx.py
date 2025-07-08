import os, sys
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import piexif
import torch
import torchvision
from sklearn.preprocessing import StandardScaler
from PIL import Image
import onnxruntime as ort

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = '/home/josh/ClotSimNet-Models/base/resnet_152/resnet_152_base.onnx'
images_dir = '/home/josh/clotsimnet/data/cnn_data_crop'
# TODO: Change target size to 512, 512 after new models are trained and exported
target_size = (512, 512) # Expected size of image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..', '..')))          
from data_modules import image_transformations    # 2️⃣ same val transform
from model_classes import EfficientNets 
val_tf = image_transformations.transform_val() 

# Load ONNX model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
    
# print(labels)
results = []
for fn in sorted(os.listdir(images_dir)):

    fullpath = os.path.join(images_dir, fn)

    # --- read the true permeability from EXIF (UserComment) ---
    pil = Image.open(fullpath)
    label_k = None
    exif_bytes = pil.info.get('exif')
    if exif_bytes:
        exif_dict = piexif.load(exif_bytes)
        user_comment = exif_dict['Exif'].get(piexif.ExifIFD.UserComment)
        if user_comment:
            # our script stored str(permeability) in UserComment
            label_k = float(user_comment.decode('utf-8'))

    # --- load pixels & apply the SAME val pipeline ---
    # decode_image → [1,H,W] torch.FloatTensor in [0..255]
    img_tensor = torchvision.io.read_image(
    fullpath, mode=torchvision.io.image.ImageReadMode.GRAY
    ).float()

    # this does Grayscale/CenterCrop(512)/ToDtype/Normalize exactly as in training
    x = val_tf(img_tensor)           # now shape [1,512,512], dtype=bfloat16
    x = x.to(torch.float32)
    x = x.unsqueeze(0)    # [B=1,C=1,H=512,W=512]
    
    # Convert to NumPy for ONNX
    x_np = x.cpu().numpy()
    
    # Run inference
    logk_np = session.run([output_name], {input_name: x_np})[0]
    pred_k = float(np.exp(logk_np.squeeze()))
    
    results.append({
        'filename':  os.path.basename(fn),
        'permeability': label_k,
        'prediction': pred_k,
    })
        
# 4) Save a CSV
pd.options.display.float_format = '{:.6e}'.format
df = pd.DataFrame(results)
print(df)
df.to_csv('resnet_152_inference.csv', index=False, float_format='%.6e')