from flow import RunFlow
import geo_to_bc_dict as bc_dict
import os
import glob
import tqdm

# Will need to specify it like this
# data_dir = '/home/joshgregory/clotsimnet/cfd/cfd_code/testing/aN_100_rp_010_seed_1'

# base_dir = '/home/joshgregory/clotsimnet/cfd/cfd_code/testing/cfd_small_2'

# base_dir = '/home/joshgregory/clotsimnet/cfd/cfd_code/testing/aN_100_rp_010_seed_1'



# h5_files = glob.glob(os.path.join(base_dir, '**', '*.h5'), recursive=True)
# cfd_dirs = list(set(os.path.dirname(file) for file in h5_files))

# cfd_dirs = '/home/joshgregory/clotsimnet/cfd/cfd_code/testing/aN_100_rp_010_seed_1'

def run_sim(D, H, flow_rate, data_dir):
    bc_dict.write_bc_dict(geo_directory=data_dir)

    run_flow = RunFlow(
        data_dir=data_dir,
        D=D,
        H=H,
        flow_rate=flow_rate
    )

    # Run CFD
    run_flow.run_cfd()


# Change D from 0.001 to 0.0001
# D = 0.0001 # mm^2/s, converted from 1*10**(-9) m^2/s
# H = 1

# Change flow rate from 1.3 * H
# flow_rate = 1.5 * H # Velocity of 1.3 mm/s taken from IVISim paper, using centerline velocity of 1300 micrometers/second 
# run_sim(D=D, H=H, flow_rate=flow_rate, data_dir='/home/joshgregory/clotsimnet/cfd/cfd_test_runs/aN_100_rp_010_seed_1')

# for directory in tqdm.tqdm(cfd_dirs):
#     # Change D from 0.001 to 0.0001
#     D = 0.0001 # mm^2/s, converted from 1*10**(-9) m^2/s
#     H = 1
    
#     # Change flow rate from 1.3 * H
#     flow_rate = 1.5 * H # Velocity of 1.3 mm/s taken from IVISim paper, using centerline velocity of 1300 micrometers/second 
#     run_sim(D=D, H=H, flow_rate=flow_rate, data_dir=directory)
#     print(f'Ran CFD at path {directory}')

