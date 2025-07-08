# Data Generation and CFD Simulations -- 2D
Josh Gregory


Last edited: June 16, 2025


Description: How to run CFD simulations for RSA code in 2D.

## Defining the Search Space

It is recommended to use the Jupyter notebook titled `search_space.ipynb` to find the ideal search space and number of data points for your situation. Once you have identified suitable ranges for $a_N$, $r_P$, and random seed values, note these values down for the input YAML file.

## Configuration YAML File

### Parameter Settings

Feel free to use any of the example .yaml files in the `data_gen/config_files/` directory. Starting from the top and going down, we have the following:

* `pore_num_low`, `pore_num_high`: Control the minimum and maximum number of pores ($a_N$) in the clot. These increment in steps of one. For example, if we set `pore_num_low: 300` and `pore_num_high: 305`, the resulting `pore_num` array that is passed to the code is:

```python
pore_num = [300, 301, 302, 303, 304, 305]
```
* `pore_radius_low` and `pore_radius_high` are the minimum and maximum radii for the clot pore radii, $r_P$. `pore_num` controls the size of this array. For example, if we have `pore_radius_low: 0.017`, `pore_radius_high: 0.024`, and `pore_num: 5`, we would have a resulting `pore_radius` array that starts at `pore_radius_low`, ends at `pore_radius_high`, and is `pore_num`-elements in size (here, it would be five elements long).

* `seed_low` and `seed_high` are the minimum and maximum random seeds that are used to seed the random clot microstructure. **NOTE**: Random seeds of 2 are currently excluded as they caused the meshing code to crash (for whatever reason), so if 2 is included anywhere between `seed_low` and `seed_high`, it is excluded. For example, if `seed_low: 1` and `seed_high: 5`, the resulting seed array would be

```python
seed = [1, 3, 4, 5]
```

### CFD/Meshing Settings

* `mesh_size`: The mesh size for every CFD simulation. 0.005 was used in each 2D simulation, which was found to balance computational efficiency and capturing sufficient clot microstructure detail.

* `diffusivity`: The diffusivity of the CT contrast agent. Value used was 0.0001 mm^2/s, which allowed for large Peclet numbers and based on literature values. See my MS Thesis for the citation (should be in chapter 2)

* `inlet_velocity`: Peak parabolic inlet velocity of blood into the clot. Was roughly based on the first IVISim paper in the methods section: https://doi.org/10.1038/s41598-023-49945-x

### File Settings

* `data_dir`: This is the path to the highest-level directory where all of the simulation results will be stored, with each simulation creating its own directory within `data_dir`.

* `sources_dir`: Directory path to where the pyTK-Pack `sources` sub-directory is located for the RSA code. For 2D, this relative path is `clotsimnet/data_gen/scripts/sources`.

* `tasks_file_name`: The file name to write all of the CFD tasks to. This will live within `data_dir`.

## Create CFD Tasks and Run Simulations

To create the tasks CSV file for simulation, run

```python
python3 write_tasks.py /path/to/yaml
```
where `/path/to/yaml` is the path to the YAML input file described earlier. This will then create a `tasks.csv` file for the simulation code to refer to later.

To run the CFD simulations, run

```python
python3 simulate.py /path/to/yaml
```
where `/path/to/yaml` is the path to the same YAML input file described earlier.

# File Descriptions
Each simulation is named in the following convention:

```
aN_{number of pores}_rp_{radius of pores}_seed_{seed value}
```
Note that the decimal for $r_P$ is removed, e.g. for a simulation where $a_N = 300$, $r_P = 0.024$, and $\text{seed} = 6$, the corresponding simulation ID string would become

```
aN_300_rp_024_seed_6
```
This string is referred as the `case_dir` within the code.

Refer to each individual file for a detailed description of how each work. `simulate.py` calls all of these files in the following rough order after reading in the `tasks` CSV file:

* `mesh.py` to mesh the given simulation case, pulling values for $a_N$, $r_P$, and the random seed from the `tasks.csv` file, saving it to the `case_dir`.
* `flow_steady`: Steady Stokes Flow and advection-diffusion reaction (ADR) implementation via [FLATiron](https://github.com/flowlabcu/FLATiron). This is where the velocity, pressure, and concentration fields are calculated.
* `paraview_convert_flowlab2.py`: Uses Python's `subprocess` module to run this file from the command line, since it require's ParaView's `pvbatch` Python command. This converts the resulting c.pvd file to a normalized concentration JPEG image.
* `crop_image_dynamic.py`: Removes an image's whitespace no matter the aspect ratio.
* `process_images.py`: Runs first-order and higher-order feature extraction on the resulting simulated image.

After each simulated image has been created, run `create_mlp.py` (located in `clotsimnet/ml/utils`) to combine all of the resulting `data.csv` files into a single CSV file for training the MLP later. Make sure that the YAML file is passed as an input argument, like this:

```bash
python3 create_mlp.py /path/to/yaml/file
```
