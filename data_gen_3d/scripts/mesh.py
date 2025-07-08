import sys
import subprocess
import os
import numpy as np
import glob
from typing import Union, Tuple

class MeshGenerator():
    def __init__(
        self, 
        radius: float, 
        mesh_size: float, 
        domain, 
        case_dir: str
    ):
        """
        Initializes the mesh generator object with the included source directories for RSA
        
        Parameters:
            None
            
        Methods:
            write_mesh(radius: float, num_pores: int, seed: int, mesh_size: float, case_dir: str, max_time_sec: float) -> tuple[bool, str, float | None]:
                Write the mesh given input parameters of particle radius, number of pores/particles, mesh size, case/simulation directory to save results, and maximum number of seconds to attempt a solution with RSA algorithm.
        """
        
        self.radius = radius
        self.mesh_size = mesh_size
        self.domain = domain
        self.case_dir = case_dir
        
    def find_max_n(self, L):
        for n in range(1, 1000):
            i = (L - 2 * self.radius * n) / (n + 1)
            if i <= 3 * self.mesh_size:
                return n - 1
        raise ValueError("Domain too small or mesh_size too large")
        
    def compute_bcc_positions(self):

        Lx, Ly, Lz = self.domain
        nx = self.find_max_n(Lx)
        ny = self.find_max_n(Ly)
        nz = self.find_max_n(Lz)

        ix = (Lx - 2 * self.radius * nx) / (nx + 1)
        iy = (Ly - 2 * self.radius * ny) / (ny + 1)
        iz = (Lz - 2 * self.radius * nz) / (nz + 1)

        ax = 2 * self.radius + ix
        ay = 2 * self.radius + iy
        az = 2 * self.radius + iz

        x_base = [ix + self.radius + i * ax for i in range(nx)]
        y_base = [iy + self.radius + j * ay for j in range(ny)]
        z_base = [iz + self.radius + k * az for k in range(nz)]

        positions = []

        # Corner spheres
        for x in x_base:
            for y in y_base:
                for z in z_base:
                    positions.append((x, y, z))

                    # Body-centered sphere
                    xc = x + ax / 2
                    yc = y + ay / 2
                    zc = z + az / 2
                    if (xc + self.radius + ix <= Lx) and (yc + self.radius + iy <= Ly) and (zc + self.radius + iz <= Lz):
                        positions.append((xc, yc, zc))

        return np.array(positions), (nx, ny, nz), (ax, ay, az), (ix, iy, iz)
        
    def write_gmsh(
        self, 
        filename: str, 
        positions
    ):
        # # --- Gmsh .geo script output ---
        with open(f'{filename}', 'w') as f:
            f.write('SetFactory("OpenCASCADE");\n')
            f.write('Mesh.MshFileVersion = 2.0;\n\n')

            # Box
            f.write(f'Box(1) = {{0, 0, 0, {self.domain[0]}, {self.domain[1]}, {self.domain[2]}}};\n\n')

            void_tags = []
            for i, (x, y, z) in enumerate(positions):
                tag = i + 2  # start after box ID 1
                void_tags.append(tag)
                f.write(f'Sphere({tag}) = {{{x}, {y}, {z}, {self.radius}}};\n')

            void_list = ', '.join(str(tag) for tag in void_tags)

            f.write('\n// --- Subtract voids ---\n')
            f.write(f'BooleanDifference{{ Volume{{1}}; Delete; }}{{ Volume{{{void_list}}}; Delete; }}\n\n')

            f.write('// --- Tag fluid volume ---\n')
            f.write('volumes() = Volume{:};\n')
            f.write('Physical Volume(\"Fluid\") = {volumes};\n\n')

            # No box surface tagging here (as requested)

            # Tag void surfaces together
            f.write('// --- Void surfaces ---\n')
            f.write('voidSurfaces[] = {};\n')
            for (x, y, z) in positions:
                rpad = self.radius + 1e-3
                f.write(
                f'voidSurfaces[] += Surface In BoundingBox{{{x - rpad}, {y - rpad}, {z - rpad}, {x + rpad}, {y + rpad}, {z + rpad}}};\n'
                )
            f.write('Physical Surface(1000) = {voidSurfaces[]};\n\n')

            f.write('Mesh.Algorithm = 6;\n')
            f.write(f'Mesh.MeshSizeMax = {self.mesh_size};\n')
    
    def write_mesh(self):
        '''
        Generates and writes mesh files for given parameters using number of particles (a_N) and radius (a_R0).
    
        Parameters:
            radius (float): Particle radius 
            mesh_size (float): Size parameter for mesh generation
            case_dir (str): Directory to save output files
            
        Returns:
            tuple: A tuple containing:
                - success (bool): Whether the mesh generation was successful
                - case_dir (str): Directory where the output files are saved
                - phi (float | None): Porosity of the generated mesh, or None if unsuccessful
        '''
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.case_dir, exist_ok=True)
        
        positions, a, i, (nx, ny, nz) = self.compute_bcc_positions()

        num_voids = len(positions)
        volume_voids = num_voids * (4/3) * np.pi * self.radius**3
        packing_fraction = volume_voids / (self.domain[0] * self.domain[1] * self.domain[2])
        packing_percentage = int(packing_fraction * 100)

        print(f'Radius: {str(self.radius)}')
        print(f'Number of voids: {str(num_voids)}')
        print(f'Packing fraction: {packing_fraction:.4f}')
        # plot_voids(positions, radius, domain)
        
        # Extract name for simulation
        base_name = os.path.basename(self.case_dir)
        
        rp_str = base_name.split('_')[-1]
        
        sim_id = f'bcc_lattice_rp_{rp_str}'
        
        geo_path = os.path.join(self.case_dir, sim_id + '.geo')

        self.write_gmsh(filename=geo_path, positions=positions)

        # Change directory and run geo2h5 command for CFD mesh later
        original_dir = os.getcwd()
        os.chdir(self.case_dir)
                
        command = 'geo2h5'
        # command = '/projects/jogr4852/FLATiron/src/flatiron_tk/scripts/geo2h5' # For Alpine
        args = ['-m', geo_path, '-d', '3', '-o', 
                os.path.join(self.case_dir, f'{sim_id}')]

        print('Running geo2h5')
        result = subprocess.run([command] + args, 
                                capture_output=True, 
                                text=True)
        
        if result.returncode != 0:
            print(f"geo2h5 command failed with error: {result.stderr}")

        # Remove all extra files
        file_extensions = ['*.xml', '*.pvd', '*.vtu', '*.msh', '*.txt']

        # Remove files with the specified extensions
        # for extension in file_extensions:
        #     for file in glob.glob(extension):
        #         os.remove(file)

        os.chdir(original_dir)
        
        h5_file_path = os.path.join(self.case_dir, f'{sim_id}' + '.h5')
        return h5_file_path
