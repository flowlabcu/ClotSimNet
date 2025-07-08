import sys
import create_mesh
import multiprocessing as mp
import numpy as np
from concurrent.futures import as_completed
import concurrent
from tqdm import tqdm
import pandas as pd
import time
import datetime
import yaml
import os

def run_mesh(radius, num_pores, seed, mesh_size, sources_dir, case_dir,):
    return create_mesh.mesh_R_N(
        radius=[radius],
        num_pores=[num_pores],
        seed=[seed],
        mesh_size=mesh_size,
        sources_dir=sources_dir,
        case_dir=case_dir
    )
  
def mesh(radius, num_pores, seed, mesh_size, case_dir, sources_dir):
    
    # TODO: Test if this is necessary here
    sys.path.append(sources_dir)
    
    num_cores = 1 # Use 1 core

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks to executor
        futures = {executor.submit(run_mesh, radius, num_pores, seed, mesh_size, sources_dir, case_dir)}
        
        for future in as_completed(futures):
            result = future.result()
            results.extend(result)
