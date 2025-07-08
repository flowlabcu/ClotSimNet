import os
import re
import sys
import json

def write_bc_dict(geo_directory: str):
    '''
    Uses the .geo file to write a dictionary of boundary conditions to be used in CFD simulations (flow_steady.py) later.
    
    Parameters:
        geo_directory: Path to the directory containing the .geo file
        
    Returns:
        None
    '''
    # Get all files in the geo_directoy
    if not os.path.exists(geo_directory):
        print('No geo directory!! Exiting...')
        sys.exit()

    geo_files = []
    for file in os.listdir(geo_directory):
        if file.endswith('.geo'):
            geo_files.append(file)

    # This does the same as above but it's harder to read. 
    # geo_files = [file for file in os.listdir(geo_directory) if file.endswith('.geo')]

    # Make directory to store boundary files
    boundary_directory = os.path.join(geo_directory, 'boundaries')
    if not os.path.exists(boundary_directory):
        os.mkdir(boundary_directory)

    for geo_file in geo_files:
        print(f'Reading {geo_file}')

        # Open the file in read binary mode ('rb') for fast!.
        with open(os.path.join(geo_directory, geo_file), 'rb') as gf: 
            # Try/excpet prevents an os error if reading a file with 
            # one line. That shouldn't happen though...
            for line in gf:
                if b'l10 =' in line:
                    boundary_line = line.decode()
                    
        '''
        Use regular expression library (built in) to find a key pattern
        in a string. Here, we want to get the content in the curly braces:
        l11 = newreg; Physical Surface(l11) = {l10, 33, 1, 4, 23, 59, 92};
        So we use finall(rawstring'<everything in curly braces>', string)
        This is also a list with length 1
        '''
        id = re.findall(r'\{[^}]*\}', boundary_line)
        # Now we map to a set of itegers of the numbers in the id list
        id_list = list(map(int, re.findall(r'\d+', id[0])))
        id_list = id_list[1:]
        
        id_list.sort() 
         
        '''
        Now we write the dictionary to some out file.
        Here, we use the geo file and remove the geo extension.
        We also will give this a custom extension just in case we need to 
        sort through a metric shit ton of files later.
        '''
        boundary_file = f'u-bcs-{geo_file[:-4]}.bounds'
        # Create the dictionary to hold boundary conditions
        boundary_conditions = {}

        # Add the pore conditions from the id_list
        for id in id_list:
            boundary_conditions[id + 1] = {"type": "dirichlet", "value": "fe.Constant((0, 0))"}
            
        # Write the dictionary to a JSON file
        file = os.path.join(boundary_directory, boundary_file)
        
        with open(file, 'w') as f:
            json.dump(boundary_conditions, f, indent=4)  # Use indent for pretty printing

        print(f'Wrote velocity boundaries to {file}')

def main(geo_directory):
    write_bc_dict(geo_directory=geo_directory)

if __name__ == '__main__':
    main()
