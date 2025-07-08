import sys
import subprocess
import os
import numpy as np
import glob
from typing import Union

class MeshGenerator():
    def __init__(
        self, 
        sources_dir: str, 
        max_time_sec: float=60
    ):
        """
        Initializes the MeshGenerator object with the included source directories for RSA
        
        Parameters:
            sources_dir (str): Directory containing the source files for files gmshUtilities.py and randSeqAdd.py, specified from the input YAML file.
            max_time_sec (float): Maximum number of seconds to try and find RSA solution.
            
        Methods:
            write_mesh(radius: float, num_pores: int, seed: int, mesh_size: float, case_dir: str, max_time_sec: float) -> tuple[bool, str, float | None]:
                Write the mesh given input parameters of particle radius, number of pores/particles, mesh size, case/simulation directory to save results, and maximum number of seconds to attempt a solution with RSA algorithm.
        """
        self.sources_dir: str = sources_dir
        self.max_time_sec: float = max_time_sec
        sys.path.append(sources_dir)
        
        global RSA, GMU
        import sources.randSeqAdd as RSA
        import sources.gmshUtilities as GMU
    
    def write_mesh(
        self, 
        radius: float, 
        num_pores: int, 
        seed: int, 
        mesh_size: float, 
        case_dir: str, 
        max_time_sec: float
    ):
        """
        Generates and writes mesh files for given parameters using number of particles (a_N) and radius (a_R0).
    
        Parameters:
            radius (float): Particle radius 
            num_pores (int): Number of particles
            seed (int): Random seed for RSA
            mesh_size (float): Size parameter for mesh generation
            case_dir (str): Directory to save output files
            max_time_sec (int): Maximum number of seconds to try and find RSA solution
            
        Returns:
            tupe: A tuple containing:
                - success (bool): Whether the mesh generation was successful
                - case_dir (str): Directory where the output files are saved
                - phi (float | None): Porosity of the generated mesh, or None if unsuccessful
        """
        # Define the dimensions of the box. Currently keeping these hard-coded as we aren't changing the area/box dimensions
        boxLower = np.array([0.0,0.0])
        boxUpper = np.array([2.0,1.0])
        box = np.array([[0.0,0.0],[2.0,1.0]])
        
        # Convert a_N and a_R0 from lists to int and float respectively
        num_pores = int(num_pores[0]) if isinstance(num_pores, list) else int(num_pores)
        radius = float(radius[0]) if isinstance(radius, list) else float(radius)

        try:
            boxRSA, success = RSA.getBoxRSA2D(
                seed=seed, 
                a_LowerBounds=boxLower, 
                a_UpperBounds=boxUpper,  
                a_N=num_pores,
                a_R0=radius,
                max_time_sec=self.max_time_sec
            )

            # Calculate porosity from boxRSA dimensions
            phi = RSA.getPorosity2D(a_LowerBounds=boxLower, a_UpperBounds=boxUpper, a_PosRSA=boxRSA)

            # Round porosity to five decimal places
            phi_round = round(phi, 5)

            boxRSANew = np.zeros((boxRSA.shape[0], boxRSA.shape[1]+1), dtype=np.float32)
            boxRSANew[:,0] = boxRSA[:,0]
            boxRSANew[:,1] = boxRSA[:,1]
            boxRSANew[:,2] = 0
            boxRSANew[:,3] = boxRSA[:,2]

            # Create case directory
            os.makedirs(case_dir, exist_ok=True)
            
            print(f'Case dir: {case_dir}')

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

            # Change directory and run geo2h5 command for CFD mesh later
            original_dir = os.getcwd()
            os.chdir(case_dir)
                    
            command = 'geo2h5'
            # command = '/projects/jogr4852/FLATiron/src/flatiron_tk/scripts/geo2h5' # For Alpine
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

            os.chdir(original_dir)
            return True, case_dir, phi

        except Exception as e:
            print(f"Error during mesh generation: {str(e)}")
            success = False
            case_dir = None
            return success, case_dir, None
        
    def generate_mesh(
        self, 
        radius: float, 
        num_pores: int, 
        seed: int, 
        mesh_size: float, 
        case_dir: str
    ):
        """
        Runs mesh generation for a single CFD case.
        
        Parameters:
            radius (float): Radius of particle
            num_pores (int): Number of pores/particles
            seed (int): Pseudo-random seed
            mesh_size (float): Domain mesh size, specified from input YAML file
            case_dir (str): Directory where the output files are saved
        
        Returns:
            None
        """
        results = []
        
        result = self.write_mesh(radius=radius, num_pores=num_pores, seed=seed, mesh_size=mesh_size, case_dir=case_dir, max_time_sec=self.max_time_sec)
        
        results.append(result)
