import pandas as pd
import time
import os
import glob
# from flow_steady import RunFlow
# from mesh_parallel_single import mesh
# import geo_to_bc_dict as bc_dict
import tqdm
import subprocess
import yaml
import sys
import crop_image_dynamic
import process_images
import sys

def test_image_creation(case_dir):

# Flowlab2 pvbatch install path:
    command = '/home/josh/ParaView-5.10.0-MPI-Linux-Python3.9-x86_64/bin/pvbatch' # For flowlab2
    # command = 'pvbatch' # For Alpine
    
    # result =subprocess.run([command, 'paraview_convert_alpine.py', case_dir], capture_output=True, text=True) # For Alpine
    
    print(f'Case dir: {case_dir}')
    
    result = subprocess.run([command, 'paraview_convert_flowlab2_local_pv.py', case_dir], capture_output=True, text=True) # For Flowlab2
    
    # print(f'Result: {result}')

    image_path = os.path.join(case_dir, 'output', os.path.basename(case_dir) + '.jpeg')
    print(f'Image path: {image_path}')
    # image_path = result.stdout.strip()
    
    # print(f'Created image at {image_path}')
    
    # Crop the jpeg to remove whitespace
    cropped_image_path = crop_image_dynamic.remove_border_whitespace(uncropped_path=image_path)
    
    # Process image and append it to data.csv
    process_images.process_image(image_path=cropped_image_path, case_dir=case_dir)
    
test_image_creation(case_dir='/home/josh/clotsimnet/test_pipeline/aN_447_rp_017_seed_1')