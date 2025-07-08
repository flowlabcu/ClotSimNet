import numpy as np
import os
import itertools
import mesh
from tqdm import tqdm
import datetime
import time
import sys
import csv

def mesh_R_N(radius, num_pores, seed, mesh_size, sources_dir, case_dir, max_time_sec=60):
    
    result = {}
    porosity = None
    radius = radius
    pores = num_pores
    
    # try:
    success, case_dir, porosity = mesh.write_mesh_R_N(
        a_R0=radius,
        a_N=num_pores,
        seed=seed,
        mesh_size=mesh_size,
        sources_dir=sources_dir,
        case_dir=case_dir,
        max_time_sec=max_time_sec
    )

    result = {
        'case_dir': case_dir,
        'porosity': porosity,
        'r_p': radius,
        'a_N': num_pores,
        'seed': seed,
        'success': success,
    }

    return result


# def write_log(results, case_dir, start_time):
#     # Ensure the save directory exists
#     if not os.path.exists(case_dir):
#         os.makedirs(case_dir)

#     ### Logging portion ###
#     log_path = os.path.join(case_dir, 'mesh_sweep_rp_aN_log.txt')
#     with open(log_path, 'w') as log:
#         log.write('Mesh sweep log for a_N (number of pores) and rp (radius of particle)\n')
#         log.write('======================================\n')

#         for result in results:
#             case_dir = result['case_dir']
#             porosity = result['porosity']
#             r_p = result['r_p']
#             a_N = result['a_N']
#             seed = result['seed']
#             success = result['success']

#             # Write mesh results to a data.csv file that will be within case_dir
#             with open(os.path.join(case_dir, 'data.csv'), 'w', newline='') as csvfile:
#                 fieldnames = ['case_dir', 'porosity', 'r_p', 'a_N', 'seed', 'success']
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerow(result)

#             # Log result
#             log.write(f'Combination: phi={porosity}, rp={r_p}, a_N={a_N}, seed={seed}\n')
#             log.write(f'Success: {success}\n')
#             if success:
#                 log.write(f'Output directory for this case: {case_dir}\n')
#             else:
#                 log.write(f'Failed to generate mesh\n')
#             log.write('-' * 50 + '\n')

#         # Record the end time
#         end_time = datetime.datetime.now()
#         duration = end_time - start_time

#         # Convert duration to hours, minutes, and seconds
#         seconds = duration.total_seconds()
#         hours, remainder = divmod(seconds, 3600)
#         minutes, seconds = divmod(remainder, 60)


#         # Summary at end of log
#         log.write('\nSummary\n')
#         log.write('======================\n')
#         total = len(results)
#         successful = sum(1 for r in results if r['success'])
#         log.write(f'Total combinations attempted: {total}\n')
#         log.write(f'Successful combinations: {successful}\n')
#         log.write(f'Failed combinations: {total - successful}\n')
#         if total > 0:
#             log.write(f'Success rate: {(successful/total)*100:.1f}%\n')
#         else:
#             log.write('Success rate: N/A - no trials run\n')
#         log.write(f'Started at: {start_time}\n\n')
#         log.write(f'\nCompleted at: {datetime.datetime.now()}\n')
#         log.write(f'Total duration: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds\n')
    