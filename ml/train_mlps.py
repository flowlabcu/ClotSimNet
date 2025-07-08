import subprocess

program_list = ['train_mlp.py', 'train_mlp_o1.py']

for program in program_list:
    subprocess.call(['python3', program])
    print(f'Finished: {program}')