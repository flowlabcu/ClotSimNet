import os

def file_cleanup(case_dir: str):
    '''
    Deletes unneeded files to keep dataset size as small as possible.
    Keeps only:
    - c_{pname}_000000.vtu files (where pname is a number)
    - c.pvd files
    - c000000.pvtu files
    - .vti files
    - .csv files

    Parameters:
        case_dir (str): Directory to simulation case
        
    Returns:
        None
    '''
    
    for root, _, files in os.walk(case_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.h5') or file.startswith('c'):
                print(f'Keeping {os.path.join(root, file)}')
            else:
                print(f'Removing {os.path.join(root, file)}')
                os.remove(os.path.join(root, file))
    
    
case_dir = '/home/joshgregory/clotsimnet/data_gen_3d/lattice_test_10/bcc_lattice_rp_06662'

file_cleanup(case_dir=case_dir)