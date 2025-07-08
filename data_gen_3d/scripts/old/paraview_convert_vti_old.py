import os
import argparse


# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', required=True, help='Path to case directory')
    return parser.parse_args()

def paraview_convert(case_dir):
    # Setup paths
        vtu_path = os.path.join(case_dir, 'output', 'c000000.vtu')
        sim_id = os.path.basename(case_dir)
        print(f'Converting {vtu_path}')
        
        # Read the VTU file
        reader = XMLUnstructuredGridReader(registrationName='reader', FileName=[vtu_path])
        reader.PointArrayStatus = ['c']
        
        # Create view
        view = GetActiveViewOrCreate('RenderView')
        
        # Create calculator for normalization
        calculator = PythonCalculator(Input=reader)
        calculator.Expression = '(c-min(c))/(max(c)-min(c))'
        
        # Create image sampling
        resampler = ResampleToImage(Input=calculator)
        
        # Save the resampled data
        vti_path = os.path.join(case_dir, 'output', f'{sim_id}.vti')
        SaveData(vti_path, proxy=resampler, 
                ChooseArraysToWrite=1,
                PointDataArrays=['result'])
        print(f'Saved to {vti_path}')
    
if __name__ == '__main__':
    args = parse_args()
    paraview_convert(args.case_dir)