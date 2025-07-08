import sys
import subprocess
import os
# sys.path.append('/home/joshgregory/clotsimnet/cfd/scripts/sources')
# import randSeqAdd as RSA
# import sources.gmshUtilities as GMU
import numpy as np
import glob

def write_mesh_R_N(a_R0, a_N, seed, mesh_size, sources_dir, case_dir, max_time_sec=60):
    """
    Generates and writes mesh files for given parameters using number of particles (a_N) and radius (a_R0).
    
    Parameters:
        seed (int): Random seed for RSA
        a_N (int): Number of particles
        a_R0 (float): Particle radius 
        mesh_size (float): Size parameter for mesh generation
        save_dir (str): Directory to save output files
    
    Returns:
        bool: True if successful, False if failed
        str: Directory where files were saved (or None if failed)
    """
    
    # TODO: Test if this is necessary here
    sys.path.append(sources_dir)
    
    import sources.randSeqAdd as RSA
    import sources.gmshUtilities as GMU

    # Define the dimensions of the box. Currently keeping these hard-coded as we aren't changing the area/box dimensions
    boxLower = np.array([0.0,0.0])
    boxUpper = np.array([2.0,1.0])
    box = np.array([[0.0,0.0],[2.0,1.0]])
    
    # Convert a_N and a_R0 from lists to int and float respectively
    a_N = int(a_N[0]) if isinstance(a_N, list) else int(a_N)
    a_R0 = float(a_R0[0]) if isinstance(a_R0, list) else float(a_R0)

    try:
        boxRSA, success = RSA.getBoxRSA2D(
            seed=seed, 
            a_LowerBounds=boxLower, 
            a_UpperBounds=boxUpper,  
            a_N=a_N,
            a_R0=a_R0,
            max_time_sec=max_time_sec
        )

        # Calculate porosity from boxRSA dimensions
        phi = RSA.getPorosity2D(a_LowerBounds=boxLower, a_UpperBounds=boxUpper, a_PosRSA=boxRSA)

        # Round porosity to five decimal places
        phi_round = round(phi, 5)

        # print(boxRSA)

        boxRSANew = np.zeros((boxRSA.shape[0], boxRSA.shape[1]+1), dtype=np.float32)
        boxRSANew[:,0] = boxRSA[:,0]
        boxRSANew[:,1] = boxRSA[:,1]
        boxRSANew[:,2] = 0
        boxRSANew[:,3] = boxRSA[:,2]

        # Create case directory
        os.makedirs(case_dir, exist_ok=True)

        # Extract simulation id from the given output directory
        sim_id = os.path.basename(case_dir)
        
        text_path = os.path.join(case_dir, f'{sim_id}.txt')
        geo_path = os.path.join(case_dir, f'{sim_id}.geo')

        # Save particle positions
        np.savetxt(text_path, boxRSANew)

        # Generate .geo file
        GMU.xyBoxPackingGeoWriterFixed(a_Box=box, a_XYZFile=text_path, a_GeoFile=geo_path, a_Sizing=mesh_size)

        # Open existing .geo file and read the contents
        with open(geo_path, 'r') as f:
            content = f.read()

        # Write the Mesh version line to the .geo file while it's still open
        with open(geo_path, 'w') as f:
            f.write('Mesh.MshFileVersion = 2.0;\n' + content)

        # print(f'Geo file written to {geo_path}')

        # Change directory and run geo2h5 command for CFD mesh later
        original_dir = os.getcwd()
        os.chdir(case_dir)
                
        command = 'geo2h5'
        # command = '/projects/jogr4852/FLATiron/src/flatiron_tk/scripts/geo2h5'
        args = ['-m', geo_path, '-d', '2', '-o', 
                os.path.join(case_dir, f'{sim_id}')]


        result = subprocess.run([command] + args, 
                                capture_output=True, 
                                text=True)

        if result.returncode != 0:
            print(f"geo2h5 command failed with error: {result.stderr}")
            os.chdir(original_dir)
            return False, case_dir, None

        # Remove all extra files
        file_extensions = ['*.xml', '*.pvd', '*.vtu', '*.msh', '*.txt']

        # Remove files with the specified extensions
        for extension in file_extensions:
            for file in glob.glob(extension):
                os.remove(file)

        # print('.msh file created')
        os.chdir(original_dir)
        return True, case_dir, phi

    except Exception as e:
        print(f"Error during mesh generation: {str(e)}")
        success = False
        case_dir = None
        return success, case_dir, None
