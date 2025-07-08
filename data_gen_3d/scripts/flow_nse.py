import sys 
import os
import time 
import json
import csv
import numpy as np

import fenics as fe 

from flatiron_tk.physics import StokesFlow
from flatiron_tk.physics import SteadyIncompressibleNavierStokes
from flatiron_tk.physics import ScalarTransport, TransientScalarTransport
from flatiron_tk.io import h5_mod
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver
from flatiron_tk.solver import NonLinearProblem, NonLinearSolver
from mpi4py import MPI

from petsc4py import PETSc
from flatiron_tk.solver import BlockNonLinearSolver
from flatiron_tk.solver import FieldSplitTree
from flatiron_tk.solver import ConvergenceMonitor

import time 

class RunFlow:
    def __init__(
        self, 
        case_dir: str, 
        D: float, 
        max_vel: int, 
        hdf5_path: str
    ):
        """
        Class to run CFD simulations from a given case directory using steady Navier-Stokes and steady ADR.
         
        Parameters:
            case_dir (str): Path to the directory for a single simulation
            D (float): Diffusivity specified from input YAML file
            H (int): Height of the rectangular domain
            max_vel (int): Max velocity specified from input YAML file
            
        Methods:                
            run_cfd():
                Run a single CFD simulation and save the results
        """
        
        # Start timer
        start = time.time()
        
        self.case_dir = case_dir
        self.D = D
        self.max_vel = max_vel
        self.hdf5_path = hdf5_path
        
        sim_id = os.path.basename(self.case_dir) 
        
        self.mesh_file = self.hdf5_path

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        self.mesh = Mesh(mesh_file=self.mesh_file)
    
    def print(self, string):
        if MPI.COMM_WORLD.rank == 0:
            sys.stdout.write(str(string) + '\n')
            sys.stdout.flush()
        
    # -------------------------------------------------------- #
    # ---- Define boundaries to mark post mesh generation ---- #
    # -------------------------------------------------------- #
    def left(self, x, left_bnd):    # Defining left boundery function
        return abs(x[0] - left_bnd) < fe.DOLFIN_EPS # Epsilon - machine precision
    
    def right(self, x, right_bnd):  # Defining right boundery function
        return abs(right_bnd - x[0]) < fe.DOLFIN_EPS
    
    def top(self, x, top_bnd):      # Defining top boundery function
        return abs(x[1] - top_bnd) < fe.DOLFIN_EPS
    
    def bottom(self, x, bottom_bnd): # Defining bottom boundery function
        return abs(bottom_bnd - x[1]) < fe.DOLFIN_EPS
    
    def front(self, x, front_bnd):  # Defining front boundery function
        return abs(x[2] - front_bnd) < fe.DOLFIN_EPS
    
    def back(self, x, back_bnd):    # Defining back boundery function
        return abs(back_bnd - x[2]) < fe.DOLFIN_EPS
    
    def set_ksp_u(self, ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setMonitor(ConvergenceMonitor("|----KSP0", verbose=False))
        ksp.setTolerances(max_it=3)
        ksp.pc.setType(PETSc.PC.Type.JACOBI)
        ksp.setUp()

    def set_ksp_p(self, ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setMonitor(ConvergenceMonitor("|--------KSP1", verbose=False))
        ksp.setTolerances(max_it=5)
        ksp.pc.setType(PETSc.PC.Type.HYPRE)
        ksp.pc.setHYPREType("boomeramg")
        ksp.setUp()

    def set_outer_ksp(self, ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setGMRESRestart(30)
        ksp.setTolerances(rtol=1e-8, atol=1e-10)
        ksp.setMonitor(ConvergenceMonitor("Outer ksp"))
        
    def run_cfd(self):
        '''
        Run a single CFD simulation and save the results.
        '''
        id_in = 1; wall_1 = 2; wall_2 = 3
        wall_3 = 4; wall_4 = 5; id_out = 6
        
        self.mesh.mark_boundary(id_in, self.left, (0.))
        self.mesh.mark_boundary(id_out, self.right, (1.))
        self.mesh.mark_boundary(wall_1, self.top, (1.))
        self.mesh.mark_boundary(wall_2, self.bottom, (0.))
        self.mesh.mark_boundary(wall_3, self.front, (0.))
        self.mesh.mark_boundary(wall_4, self.back, (1.))
        
        start = time.time()
        
        mu = 4.3e-6
        rho = 1.06e-6
        nu = mu / rho
        U = self.max_vel
        
        nse = SteadyIncompressibleNavierStokes(self.mesh)
        nse.set_element('CG', 1, 'CG', 1)
        nse.build_function_space()
        nse.set_dynamic_viscosity(mu)
        nse.set_density(rho)
        nse.set_weak_form()
        nse.add_stab()
        
        # Void ID 
        voids = 1000
        
        # pore_size = os.path.basename(self.case_dir).split('_')[-1]  # Gets "03000"
        
        # print(f'Pore size: {str(pore_size)}')

        # # Insert decimal point after first zero
        # pore_formatted = pore_size[0] + '.' + pore_size[1:]  # Makes "0.3000"


        # Re = float(pore_formatted) * rho * U / mu
        # print(f'Reynolds number: {Re}')
        
        one = fe.interpolate(fe.Constant(1), nse.V.sub(1).collapse())
        area = fe.assemble(one * nse.dx) 
        
        center = [0, 0.5, 0.5] # Coordinates of the inlet face in x,y,z
        flow_direction = [1, 0, 0] # Inlet flow direction, x-axis only
        face_radius = 1 / np.sqrt(2) # Ensuring that the circle radius is equal to the diagonal radius of the square inlet
        
        inlet_paraboloid = fe.Expression(
                                (
                                'u * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) / (R * R) * nx',
                                'u * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) / (R * R) * ny',
                                'u * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) / (R * R) * nz'
                                ),
                                degree=2, u=U, R=face_radius,
                                xc0=center[0], xc1=center[1], xc2=center[2],
                                nx=flow_direction[0], ny=flow_direction[1], nz=flow_direction[2]
                                )

        u_bcs = {
                id_in: {'type': 'dirichlet', 'value': inlet_paraboloid},
                wall_1: {'type': 'dirichlet', 'value': fe.Constant((0, 0, 0))},
                wall_2: {'type': 'dirichlet', 'value': fe.Constant((0, 0, 0))},
                wall_3: {'type': 'dirichlet', 'value': fe.Constant((0, 0, 0))},
                wall_4: {'type': 'dirichlet', 'value': fe.Constant((0, 0, 0))},
                voids: {'type': 'dirichlet', 'value': fe.Constant((0, 0, 0))},
                }

        p_bcs = {id_out: {'type': 'dirichlet', 'value': fe.Constant(0)}}
        bc_dict = {'u': u_bcs, 'p': p_bcs}
        nse.set_bcs(bc_dict)
        nse.set_writer(os.path.join(self.case_dir, 'output'), 'pvd')
        
        split = {'fields': ('u', 'p'),
            'composite_type': 'schur',
            'schur_fact_type': 'full',
            'schur_pre_type': 'a11',
            'ksp0_set_function': self.set_ksp_u,
            'ksp1_set_function': self.set_ksp_p}

        problem = NonLinearProblem(nse)
        tree = FieldSplitTree(nse, split)
        
        solver = BlockNonLinearSolver(
            tree, 
            self.mesh.comm, 
            problem, 
            fe.PETScKrylovSolver(), outer_ksp_set_function=self.set_outer_ksp
        )

        solver.solve()
        nse.write()
        
        (u, p) = nse.solution_function().split(deepcopy=True)

        stop = time.time()

        print('CFD completed')
        print(f'Time taken: {stop - start} seconds')
        
        # -------------------------------------------------------- #
        # ---------------- Set up Scalar Transport --------------- #
        # -------------------------------------------------------- #
        tsp = ScalarTransport(self.mesh)
        tsp.set_element('CG', 1)
        tsp.build_function_space()
        (u, _) = fe.split(nse.solution)
        tsp.set_advection_velocity(u)
        tsp.set_diffusivity(self.D)
        tsp.set_reaction(0.)
        tsp.set_weak_form()
        tsp.add_stab()
        tsp.set_tag('c')
        
        total_flux = fe.Expression(('u * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) / (R * R) * nx'),
                                degree=2, u=1.0, R=face_radius,
                                xc0=center[0], xc1=center[1], xc2=center[2],
                                nx=flow_direction[0])
        advective_flux = tsp.solution_function() * tsp.get_advection_velocity()
        
        c_bcs = {
                    id_in: {'type': 'neumann', 'value': total_flux * fe.Constant((-1.0, 0.0, 0.0)) + advective_flux},
                    voids: {'type': 'dirichlet', 'value': fe.Constant(0.0)},
                }
        tsp.set_bcs(c_bcs)
        tsp.set_writer(os.path.join(self.case_dir, 'output'), 'pvd')
        la_solver_tsp = fe.LUSolver()
        solver_tsp = PhysicsSolver(tsp, la_solver=la_solver_tsp)
        solver_tsp.solve()
        tsp.write()
        
        print('ADR completed')
        
        # Calculate permeability with Darcy's Law
        width = 1.0
        height = 1.0
        length = 1.0
        
        u_avg_x = fe.assemble(u[0] * nse.dx) / length
        u_avg_y = fe.assemble(u[1] * nse.dx) / width
        u_avg_z = fe.assemble(u[2] * nse.dx) / height
        u_avg = np.sqrt(u_avg_x**2 + u_avg_y**2 + u_avg_z**2)

        delta_p = (fe.assemble(p * nse.ds(id_out)) - fe.assemble(p * nse.ds(id_in))) / width**2
        delta_x = length
        
        Q = u_avg * (width*height) # Q = u * A

        k = -Q * mu / (delta_p / delta_x)

        data_dict = {
            'id': self.case_dir,
            'Q': Q,
            'dP': delta_p,
            'k': k,
            'u_mean': u_avg,
        }
        
        # Path to the data file
        data_file = os.path.join(self.case_dir, 'data.csv')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(data_file), exist_ok=True)

        # Check if the file exists
        if not os.path.isfile(data_file):
            # If it doesn't exist, create it and write the header
            with open(data_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_dict.keys())
                writer.writeheader()
            existing_data = []  # No existing data

        else:
            # Read the existing data
            with open(data_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_data = [row for row in reader]  # Read rows into a list of dictionaries
                fieldnames = reader.fieldnames

        # Add the new data to existing rows
        for row in existing_data:
            row.update(data_dict)

        # Append the new data if it's not in the file
        if data_dict not in existing_data:
            existing_data.append(data_dict)

        # Write the updated data back to the file
        with open(data_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            writer.writeheader()
            writer.writerows(existing_data)

        print(f"Wrote data to {data_file}.")
 