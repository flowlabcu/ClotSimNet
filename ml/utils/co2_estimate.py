import pandas as pd
import numpy as np

"""
co2_estimate.py

Script to estimate the amount of CO2 released during training. CSV file is downloaded from WandB for the power curve during training.

The power curve is integrated over time to obtain the energy used, then the splits for different energy sources are applied. Energy sources are from Andy Monaghan of the CU Research Computing Support Team. Current as of June 2025 for the Boulder, CO area.
"""

# Path to power curve CSV file downloaded from WandB
df = pd.read_csv('/home/josh/clotsimnet/ml/utils/gh200_power_ex.csv')

time = df.iloc[:, 0]
power = df.iloc[:, 1]

energy = np.trapz(y=power, x=time)

energy_kwhr = energy * (2.778*10**(-7))

# Most recent stats from Xcel say CO 

coal = 0.19
nat_gas = 0.3
nuclear = 0.1
wind = 0.32
solar = 0.04
other_renew = 0.04


coal_kwhr = 2.21 # From Andy
nat_gas_kwhr = 0.91 # From Andy

coal_energy = energy_kwhr * coal # Get fraction of energy that was produced with non-renewable sources
nat_gas_energy = energy_kwhr * nat_gas

coal_co2 = coal_energy * coal_kwhr
nat_gas_co2 = nat_gas_energy * nat_gas_kwhr

total = coal_co2 + nat_gas_co2

print(f'CO2 emissions: {total} pounds')
print(f'CO2 emissions: {total*0.4536} kg')