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

class RunFlow:
    def __init__(self, case_dir: str, D: float, H: int, max_vel: int):
        '''
        Class to run CFD simulations from a given case directory.
         
        Parameters:
            case_dir (str): Path to the directory for a single simulation
            D (float): Diffusivity specified from input YAML file
            H (int): Height of the rectangular domain
            max_vel (int): Max velocity specified from input YAML file
            
        Methods:                
            run_cfd():
                Run a single CFD simulation and save the results
        '''
        
        # Start timer
        start = time.time()
        
        self.case_dir = case_dir
        self.D = D
        self.H = H
        self.max_vel = max_vel
        
        case_name = os.path.basename(case_dir)

        self.mesh_file = os.path.join(case_dir, case_name + '.h5')
        
        self.u_boundary_file = os.path.join(case_dir, 'boundaries', f'u-bcs-{case_name}.bounds')
        
        self.mesh = Mesh(mesh_file = self.mesh_file)
        
    def set_ksp_u(self, ksp):
            ksp.setType(PETSc.KSP.Type.FGMRES)
            ksp.setMonitor(ConvergenceMonitor("|----KSP0", verbose=True))
            ksp.setTolerances(rtol=1e-6, atol=1e-8, max_it=50)
            # ksp.pc.setType(PETSc.PC.Type.JACOBI)
            ksp.pc.setType(PETSc.PC.Type.HYPRE)
            ksp.pc.setHYPREType("boomeramg")
            ksp.setUp()

    def set_ksp_p(self, ksp):
        ksp.setType(PETSc.KSP.Type.FGMRES)
        ksp.setMonitor(ConvergenceMonitor("|--------KSP1", verbose=True))
        ksp.setTolerances(max_it=30)
        ksp.pc.setType(PETSc.PC.Type.HYPRE)
        ksp.pc.setHYPREType("boomeramg")
        ksp.setUp()

    def set_outer_ksp(self, ksp):
        ksp.setType(PETSc.KSP.Type.BCGS)
        ksp.pc.setType(PETSc.KSP.Type.HYPRE)
        ksp.pc.setHYPREType('boomeramg')
        # ksp.setGMRESRestart(30)
        ksp.setTolerances(rtol=1e-6, atol=1e-8, max_it=5000)
        ksp.setMonitor(ConvergenceMonitor("Outer ksp"))
        
    def run_cfd(self):
        '''
        Run a single CFD simulation and save the results.
        '''
        
        start = time.time()
        
        id_in = 4000000
        id_out = 2000000

        # Only unique values are for inlet/outlet, rest are walls (don't care which ones are which, same BCs for all)
        wall_1 = 1000000
        wall_2 = 3000000
        wall_3 = 5000000
        wall_4 = 6000000


        # msh = Mesh(mesh_file='aN_447_rp_01700_seed_1.h5')

        # Define problem object (Steady incomp. Navier Stokes)
        nse = SteadyIncompressibleNavierStokes(self.mesh)
        nse.set_element('CG', 1, 'CG', 1)
        nse.build_function_space()
        # mu = 4.3e-6 # m / s; I'm inlcuding both so we can get permeability
        # nu = 4.0566e-6 # m^2 / s
        
        mu = 4e-9 # N*s/mm^2
        nu = 3.77358 # mm^2 / s
        rho = 1.06e-6 # kg/mm^3
        
        nse.set_dynamic_viscosity(mu)
        nse.set_density(rho)
        nse.set_weak_form()
        nse.add_stab()
        
        # Define problem object (transport)
        theta = 0.5
        dt = 0.01
        tsp = ScalarTransport(self.mesh)
        tsp.set_element('CG', 1)
        tsp.build_function_space()
        (u, p) = fe.split(nse.solution)
        tsp.set_advection_velocity(u)
        tsp.set_diffusivity(self.D)
        tsp.set_reaction(0.)
        tsp.set_weak_form()
        tsp.add_stab()
        tsp.set_tag('c')


        one = fe.interpolate(fe.Constant(1), nse.V.sub(1).collapse())
        area = fe.assemble(one * nse.dx)  
        # U = self.get_velocity(q=flow_rate_in, H=self.H)

        center = [0, 0.5, 0.5] # Coordinates of the inlet face in x,y,z
        flow_direction = [1, 0, 0] # Inlet flow direction, x-axis only
        face_radius = 1 / np.sqrt(2) # Ensuring that the circle radius is equal to the diagonal radius of the square inlet

        U = self.max_vel # Pull max. centerline velocity from __init__

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
        wall_4: {'type': 'dirichlet', 'value': fe.Constant((0, 0, 0))}
        }

        with open(self.u_boundary_file, 'r') as file:
            u_bcs.update(json.load(file))
            u_bcs = {int(k): v for k, v in u_bcs.items()}

        for key, value in u_bcs.items():
            if value["value"] == "fe.Constant((0, 0, 0))":
                value["value"] = fe.Constant((0, 0, 0))
                
        C = 1 # Match 2D case
                
        conc_paraboloid = fe.Expression(
        (
        'c * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) * (R - sqrt(pow(x[0] - xc0, 2) + pow(x[1] - xc1, 2) + pow(x[2] - xc2, 2))) / (R * R) * nx'
        ),
        degree=2, c=C, R=face_radius,
        xc0=center[0], xc1=center[1], xc2=center[2],
        nx=flow_direction[0], ny=flow_direction[1], nz=flow_direction[2]
        )
        
        # Concentrations
        total_flux = conc_paraboloid
        
        advective_flux = tsp.solution_function() * tsp.get_advection_velocity()
        
        ### ---- Specifying different concentration boundary conditions here, labeled 1-3 ---- ###
        
        # 1. Total flux w/ advective flux
        c_bcs = {
            id_in: {'type': 'neumann', 'value': total_flux * fe.Constant((-1.0, 0.0, 0.0)) + advective_flux} # Want inward normal, hence -1.0 instead of 1.0
        }
        
        # print('poooooooooop')
        # import sys
        # sys.exit()
        # print('No cancel')
        
        # 2. Fixed at inlet
        # c_bcs = {
        #     id_in: {'type': 'dirichlet', 'value': fe.Constant(1.0)}
        # }
        
        # 3. Diffusive flux
        # c_bcs = {
        #     id_in: {'type': 'neumann', 'value': fe.Constant((-1.0, 0.0))} 
        # }
        
        
        with open(self.u_boundary_file, 'r') as file:
            c_bcs.update(json.load(file))
            c_bcs = {int(k): v for k, v in c_bcs.items()}
            
        # Replace string "fe.Constant((0, 0))" with the actual value
        for key, value in c_bcs.items():
            if value["value"] == "fe.Constant((0, 0, 0))":
                value["value"] = fe.Constant(0)

        p_bcs = {id_out: {'type': 'dirichlet', 'value': fe.Constant(0)}}
        bc_dict = {'u': u_bcs, 'p': p_bcs}

        nse.set_bcs(bc_dict)
        tsp.set_bcs(c_bcs)
        
        # Set output writer (folder, file ext)
        nse.set_writer(os.path.join(self.case_dir, 'output'), 'pvd')
        tsp.set_writer(os.path.join(self.case_dir, 'output'), 'pvd')

        split = {'fields': ('u', 'p'),
                    'composite_type': 'schur',
                    'schur_fact_type': 'full',
                    'schur_pre_type': 'selfp',
                    'ksp0_set_function': self.set_ksp_u,
                    'ksp1_set_function': self.set_ksp_p}
        
        # Set solver and solve
        problem = NonLinearProblem(nse)
        tree = FieldSplitTree(nse, split)
        solver = BlockNonLinearSolver(tree, self.mesh.comm,
                                    problem, fe.PETScKrylovSolver(),
                                    outer_ksp_set_function=self.set_outer_ksp)

        solver.solve()
        nse.write()
        
        la_solver_tsp = fe.LUSolver()
        solver_tsp = PhysicsSolver(tsp, la_solver=la_solver_tsp)
        
        solver_tsp.solve()
        tsp.write()
        
        width = 1.0
        height = 1.0
        length = 2.0
        
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