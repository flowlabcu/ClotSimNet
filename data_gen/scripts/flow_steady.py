import sys 
import os
import time 
import json
import csv
import numpy as np

import fenics as fe 

from flatiron_tk.physics import StokesFlow
from flatiron_tk.physics import ScalarTransport, TransientScalarTransport
from flatiron_tk.io import h5_mod
from flatiron_tk.mesh import Mesh
from flatiron_tk.solver import PhysicsSolver
from mpi4py import MPI


class RunFlow:
    def __init__(
        self, 
        case_dir: str, 
        D: float, 
        H: int, 
        flow_rate: int
    ):
        """
        Class to run CFD simulations from a given case directory.
         
        Parameters:
            case_dir (str): Path to the directory for a single simulation
            D (float): Diffusivity specified from input YAML file
            H (int): Height of the rectamgular domain
            flow_rate (int): Flow rate specified from input YAML file
            
        Methods:
            get_velocity(q: int, H: int) -> float:
                Calculate velocity from flow rate and height of rectangular domain
                
            run_cfd():
                Run a single CFD simulation and save the results
        """
        # Start timer
        start = time.time()
        
        self.case_dir = case_dir
        self.D = D
        self.H = H
        self.flow_rate = flow_rate
        
        case_name = os.path.basename(case_dir)

        self.mesh_file = os.path.join(case_dir, case_name + '.h5')

        self.u_boundary_file = os.path.join(case_dir, 'boundaries', f'u-bcs-{case_name}.bounds')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        self.mesh = Mesh(mesh_file = self.mesh_file)
        
        self.n = self.mesh.facet_normal()

    def get_velocity(
        self, 
        q: int, 
        H: int
    ) -> float:
        """
        Calculate velocity from flow rate and height of rectangular domain/
        
        Parameters:
            q (int): Flow rate
            H (int): Height of rectantular domain
            
        Returns:
            u (float): Velocity
        """
        u = (6*q)/H
        return u

    def run_cfd(self):
        '''
        Run a single CFD simulation and save the results.
        '''
        
        start = time.time()

        id_in = 4000000
        id_out = 2000000
        id_bot = 3000000
        id_top = 1000000
        
        # Define problem object (stokes)
        stk = StokesFlow(self.mesh)
        stk.set_element('CG', 1, 'CG', 1)
        stk.build_function_space()
        # mu = 4.3e-6 # m / s; I'm inlcuding both so we can get permeability
        # nu = 4.0566e-6 # m^2 / s
        mu = 4e-9 # N*s/mm^2
        nu = 3.77358 # mm^2 / s
        
        stk.set_kinematic_viscosity(nu)
        stk.set_weak_form()
        stk.add_stab()

        # Define problem object (transport)
        theta = 0.5
        dt = 0.01
        tsp = ScalarTransport(self.mesh)
        tsp.set_element('CG', 1)
        tsp.build_function_space()
        (u, p) = fe.split(stk.solution)
        tsp.set_advection_velocity(u)
        tsp.set_diffusivity(self.D)
        tsp.set_reaction(0.)
        tsp.set_weak_form()
        tsp.add_stab()
        tsp.set_tag('c')

        # READ IN DICTIONARIES
        flow_rate_in = self.flow_rate

        one = fe.interpolate(fe.Constant(1), stk.V.sub(1).collapse())
        area = fe.assemble(one * stk.dx)  
        U = self.get_velocity(q=flow_rate_in, H=self.H)
        
        u_bcs = {
        id_in: {'type': 'dirichlet', 'value': fe.Expression(('U * x[1]*(H-x[1])/(H*H)','0'), U=U, H=self.H, t=0, degree=2)},
        id_top: {'type': 'dirichlet', 'value': fe.Constant((0, 0))},
        id_bot: {'type': 'dirichlet', 'value': fe.Constant((0, 0))}
        }

        
        with open(self.u_boundary_file, 'r') as file:
            u_bcs.update(json.load(file))
            u_bcs = {int(k): v for k, v in u_bcs.items()}
            

        # Replace string "fe.Constant((0, 0))" with the actual value
        for key, value in u_bcs.items():
            if value["value"] == "fe.Constant((0, 0))":
                value["value"] = fe.Constant((0, 0))

        # Concentrations
        total_flux = fe.Expression('C * x[1]*(H-x[1])/(H*H)', C=1, H=self.H, t=0, degree=2)
        
        advective_flux = tsp.solution_function() * tsp.get_advection_velocity()
        
        ### ---- Specifying different concentration boundary conditions here, labeled 1-3 ---- ###
        
        # 1. Total flux w/ advective flux
        c_bcs = {
            id_in: {'type': 'neumann', 'value': total_flux * fe.Constant((-1.0, 0.0)) + advective_flux} # Want inward normal, hence -1.0 instead of 1.0
        }
        
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
            if value["value"] == "fe.Constant((0, 0))":
                value["value"] = fe.Constant(0)

        p_bcs = {id_out: {'type': 'dirichlet', 'value': fe.Constant(0)}}
        bc_dict = {'u': u_bcs, 'p': p_bcs}

        stk.set_bcs(bc_dict)
        tsp.set_bcs(c_bcs)
        

        # Set output writer (folder, file ext)
        stk.set_writer(os.path.join(self.case_dir, 'output'), 'pvd')
        tsp.set_writer(os.path.join(self.case_dir, 'output'), 'pvd')

        # Set solver and solve
        la_solver = fe.LUSolver()
        solver = PhysicsSolver(stk, la_solver)

        solver.solve()
        stk.write()
        (u, p) = stk.solution_function().split(deepcopy=True)

        la_solver_tsp = fe.LUSolver()
        solver_tsp = PhysicsSolver(tsp, la_solver=la_solver_tsp)

        solver_tsp.solve()
        tsp.write()

        width = 1.0
        length = 2.0
        
        u_avg_x = fe.assemble(u[0] * stk.dx) / length
        u_avg_y = fe.assemble(u[1] * stk.dx) / width
        u_avg = np.sqrt(u_avg_x**2 + u_avg_y**2)

        delta_p = (fe.assemble(p * stk.ds(id_out)) - fe.assemble(p * stk.ds(id_in))) / width
        delta_x = length
        
        Q = u_avg * width

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
