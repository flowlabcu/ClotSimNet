import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import piexif
import torchvision

tensor_dtype = torch.bfloat16

# Dynamically construct the path based on the user's home directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..', '..')))
from model_classes import EfficientNets
from data_modules import image_transformations, load_data

# 1) Model and device
TS_MODEL_PATH = '/home/josh/ClotSimNet-Models/models/base/enet_b7/enet_b7_base_tscript.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load(TS_MODEL_PATH, map_location=device)
model.eval()

# 2) Where are your images?
#    If you have raw scans:
UNCROPPED_DIR = '/home/josh/clotsimnet/data/cnn_data_crop'
#    Or if you’ve already center‑cropped them 512×512:
CENTER_CROPPED_DIR = '/home/josh/clotsimnet/data/cnn_data_center_crop'

# Choose one:
USE_UNCROPPED = True
images_dir = UNCROPPED_DIR

# 3) Grab your exact validation transform
val_tf = image_transformations.transform_val()

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
    x = x.unsqueeze(0).to(device)    # [B=1,C=1,H=512,W=512]

    # --- inference ---
    with torch.no_grad():
        logk = model(x)              # network spits out log(k)
        pred_k = float(logk.exp().cpu())

    results.append({
        'filename': fn,
        'permeability': label_k,
        'prediction': pred_k,
    })

# 4) Save a CSV
pd.options.display.float_format = '{:.6e}'.format
df = pd.DataFrame(results)
print(df)
df.to_csv('ts_inference_results.csv', index=False, float_format='%.6e')
