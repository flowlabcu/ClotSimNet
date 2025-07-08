import pandas as pd
import time
import os
import shutil
import numpy as np
from flow_steady import RunFlow
import geo_to_bc_dict as bc_dict
import subprocess
import yaml
import sys
# import crop_image_dynamic
# import process_images
from mesh import MeshGenerator
import sys
sys.path.append('/home/joshgregory/clotsimnet/data_gen_3d/scripts/sources')
import sources.randSeqAdd as RSA
import sources.gmshUtilities as GMU

num_pores = 10
radius = 0.017
seed = 1
max_time_sec = float(1000)
mesh_size = 0.01

case_dir = f'/home/joshgregory/clotsimnet/data_gen_3d/aN_{num_pores}_rp_{str(radius).replace(".", "")}_seed_{seed}'

# Create case directory
os.makedirs(case_dir, exist_ok=True)

print(f'Case dir: {case_dir}')

# Extract simulation id from the given output directory
sim_id = os.path.basename(case_dir)
print(f'Sim id: {sim_id}')

boxLower = np.array([0.0, 0.0, 0.0])
boxUpper = np.array([2.0, 1.0, 1.0])
box = np.array([[0.0, 0.0, 0.0],[2.0, 1.0, 1.0]])

boxRSA, success = RSA.getBoxRSA3D(
                seed=seed, 
                a_LowerBounds=boxLower, 
                a_UpperBounds=boxUpper,  
                a_N=num_pores,
                a_R0=radius,
                max_time_sec=max_time_sec
            )

phi = RSA.getPorosity3D(a_LowerBounds=boxLower, a_UpperBounds=boxUpper, a_PosRSA=boxRSA)

print(f'Porosity: {phi}')
print(f'Corrected porosotiy: {1-phi}')

text_path = os.path.join(case_dir, f'{sim_id}.txt')
geo_path = os.path.join(case_dir, f'{sim_id}.geo')

# Save particle positions
np.savetxt(text_path, boxRSA)

# Generate .geo file
GMU.xyzBoxPackingGeoWriterFixed(a_Box=box, a_XYZFile=text_path, a_GeoFile=geo_path, a_Sizing=mesh_size)

# Open existing .geo file and read the contents
with open(geo_path, 'r') as f:
    content = f.read()

# Write the Mesh version line to the .geo file while it's still open
with open(geo_path, 'w') as f:
    f.write('Mesh.MshFileVersion = 2.0;\n' + content)

# Change directory and run geo2h5 command for CFD mesh later
original_dir = os.getcwd()
os.chdir(case_dir)

print('Running geo2h5 command')
command = 'geo2h5'
# command = '/projects/jogr4852/FLATiron/src/flatiron_tk/scripts/geo2h5' # For Alpine
args = ['-m', geo_path, '-d', '3', '-o', 
        os.path.join(case_dir, f'{sim_id}')]

result = subprocess.run([command] + args, 
                        capture_output=True, 
                        text=True)

if result.returncode != 0:
                print(f"geo2h5 command failed with error: {result.stderr}")
                os.chdir(original_dir)