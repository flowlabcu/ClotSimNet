import os
import glob

def main(data_dir: str):
    '''
    File to start at highest-level data_dir directory and walks through all subsequent directories, only keeping the .h5 mesh files needed for CFD simulations.
    
    Parameters:
    
        - data_dir: Path to the highest-level directory that contains all subdirectories with meshes.
        
    Returns:
        None
    '''
    
    for path, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith('.h5'):
                print(f'Keeping file: {os.path.join(path, name)}')
            else:
                # print(f'Would delete file: {os.path.join(path, name)}')
                os.remove(os.path.join(path, name))
                
if __name__ == '__main__':
    main(data_dir='/mnt/ssd1/joshgregory/clotsimnet_3d_100')