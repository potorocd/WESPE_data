# WESPE data (consecutive linear script)

## Introduction
This is the ancestor of the WESPE data viewer program with GUI. It possesses most features of the newer software, including extra features like a run corrector. However, the main disadvantage is that this is a consecutive linear script that generates one image per run, making all calculations (like the generation of the delay-energy map via numerous iterations) every time. As a result, an object-oriented approach was developed, which yielded the WESPE data viewer. Still, the current script can be efficiently used to explore time-resolved XPS data from the WESPE endstation.

## Example plot
<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data/blob/main/Example_plot.png" alt="Example plot" width="500"/>
</p>

## How to use	
Anaconda distribution in combination with Spyder environment is recommended.

This script works on the principle of ‘one execution – one matplotlib figure’. The figure configuration is determined by parameters set in approximately the first 200 lines of the script. One to three plots can be generated in the final figure. The first one is related to the 2D delay-energy map, where every horizontal line corresponds to a single energy-dispersive curve corresponding to a specific delay-stage value. The second plot is related to cuts across the time axis (horizontal cuts on the delay-energy map), and the third one is related to cuts across the energy axis (vertical cuts on the delay-energy map).  Including or excluding each of the three plots from the final figure is possible using one of three on/off switches.
```python
'''Graph selection'''
# Select graphs you want to see
delay_energy_map = 'on'  # Intensity versus energy and delay time both
delay_cut = 'on'  # Intensity versus energy plot (at fixed delay)
energy_cut = 'off'  # Intensity versus delay time (at fixed energy)
```

### Section I (Input selection)
This section serves for the specification of the main input parameters like: 1) where to find the data; 2) what exact hdf5 files want to load; 3) binning size for the energy and time domain; 4) selection of delay stage values as y-axis or switching to delay relative time zero; 5) cut positions and widths for the second and third plots responsible for x and y slicing of the delay-energy map.
```python
file_dir = 'D:/Data/Extension_2021_final'  # data folder
# Here we specify run numbers as a list.
run_numbers = [37378, 37379, 37380, 37381, 37382, 37383]
energy_step = 0.075  # (eV) Determines the step on the energy axis.
delay_step = 0.1  # (ps) Determines the step on the delay axis.
DLD = 'DLD4Q'  # Options: 'DLD4Q' or 'DLD1Q'
t0 = 1328.2  # For recalculation of the delay axis.
time_0 = 'off'  # Recalculation happens when time_0 = 'on'.
dif_map = 'off'  # Plotting of difference map: 'on' or 'off'. if time_0 == 'on'
t = [-0.5, 0.1, 0.5, 1.5, 3]  # Here we specify delay cut positions as a list. [t1, t2, t3]
dt = [0.2, 0.2, 0.2, 0.2]  # Here we specify delay cut widths as a list. [dt1, dt2, dt3]
e = [202.3, 202.7, 203.2]  # Here we specify energy cut positions as a list.
de = [0.3, 0.3, 0.3]  # Here we specify energy cut widths as a list.
static = 1001  # Here we specify the del. val. where static runs will be stored
```

### Section II (Region of interest selection)
This section serves to cut off electrons with specific parameters from the counting process. It allows filtering:
1. Delay values (y-axis of the delay-energy map)
2. Energy values (x-axis of the delay-energy map)
3. MacroBunch ID values
4. MicroBunch ID values  

Four corresponding on/off switches are available in the sections. When ‘on’, [min, max] parameters are taken from corresponding lists.
```python
ROI_d = 'off'  # If 'on', the filter of delay values is activated.
ROI_e = 'off'  # If 'on', the filter of energy values is activated.
B_sorting = 'off'  # If 'on', the macrobunch filter is activated.
MB_sorting = 'off'  # If 'on', the microbunch filter is activated.
# Here we specify the delay and energy region margins as two lists.
delay_range = [-0.8, 3.5]  # delay range in ps
energy_range = [198, 206]  # kinetic energy range in eV
B_range = [0, 99]  # macrobunch range in %: [0, 50] means first 50% of bunches
MB_range = [1, 440]  # microbunch range from 1 to 440
cut_t_boundaries = 'off'  # removes the highest & the lowest delay stage values 
```

### Section III (Data normalization)
The following section has three on/off switches related to each of the three possible plots (2D map, t cuts – horizontal cuts, and e cuts – vertical cuts) for various kinds of normalization.
```python
total_electron_norm = 'on'  # Divides the whole line by the total number of
                            # detected electrons.
t_cut_norm_bin = 'off'  # Data normalization for delay cuts (div. by bin size)
e_cut_norm_bin = 'off'  # Data normalization for energy cuts (div. by bin size)

map_norm = 'off'  # Data normalization for delay scan map [0,1]: 'on' or 'off'
t_cut_norm = 'off'  # Data normalization for delay cuts [0,1]:'on'/'off'
e_cut_norm = 'off'  # Data normalization for energy cuts [%]:'on'/'off'

map_norm_dif = 'off'  # Data normalization for difference map: 'on' or 'off'
t_cut_norm_dif = 'off'  # Data normalization for delay cuts [-1,1]:'on'/'off'
e_cut_norm_dif = 'off'  # Data normalization for energy cuts [-1,1]:'on'/'off'
```

### Additional features
One can switch between 2D and 3D projection for the 2D delay-energy map.
```python
map_projection = '2D'  # '2D' or '3D' options available
elev = 80  # The elevation angle in the vertical plane in degrees.
azim = 270  # The azimuth angle in the horizontal plane in degrees.
```
Besides, cuts across the time axis have two additional features. The first one allows adding difference curves to the plot, and the second one introduces a vertical offset (waterfall plot).
```python
# add difference curve to the delay cut plot
add_dif_delay_cut = 'off'  # Set 'on' if you want to get a difference plot
magn = 5  # Coefficient for multiplication of difference curves
# add vertical offset the delay cut plot (waterfall plot)
waterfall_delay_cut = 'off'  # Set 'on' to get >=0 difference between 2 curves
waterfall_offet = 0.1  # Set > 0 to increase relative shift of the curves
```

### Settings
And the last configuration section is dedicated to settings, where one can finely adjust plenty of different parameters of the resulting matplotlib figure.
```python
'''General graph parameters'''
font_family = 'Garamond'  # Figure font
font_size = 18  # Figure main font size
font_size_axis = 28  # Figure font size for axis labels
font_size_large = 32  # Figure font size for titles
dpi = 600  # Figure resolution
fig_width = 10  # Figure width (individual graph)
fig_height = 5  # Figure height (individual graph)
axes_linewidth = 1.1  # Axes linewidth (graph frames)

'''Delay-energy map parameters'''
# Tick parameters
map_n_ticks_x = 10  # Tick number for x axis (+-1 precision)/rounding to units
map_n_ticks_y = 8  # Tick number for y axis (+-1 precision)
map_n_ticks_z = 8  # Tick number for z axis (+-1 precision)
map_n_ticks_minor = 5  # Number of minor ticks
map_tick_length = 6  # Length of major ticks
# Time zero line parameters
t0_line = 'on'  # Indication of t0 for delay-energy map
line_type_t0_line = '--'  # Line type for t0 line
line_width_t0_line = 2  # Linewidth for t0 line
line_op_t0_line = 50  # Opacity for t0 line
color_t0_line = 'black'  # Color for t0 line

'''Delay cut plot parameters'''
# Main parameters
line_type_d = 'o-'  # Line type for delay cut ('-','--','o-', ':', '-.')
marker_size_d = 3.5  # Marker size for delay cut
line_width_d = 0.75  # Linewidth for delay cut
line_op_d = 70  # Line opacity for delay cut (0-100%)
# Integration area indication
int_area_d = 'on'  # Indication of integration area for delay cut
line_type_int_area_d = '-'  # Line type for delay cut integration area
line_width_int_area_d = 1.5  # Linewidth for delay cut integration area
line_op_int_area_d = 80  # Opacity for delay cut integration area (0-100%)
# Tick parameters
t_n_ticks_x = 10  # Tick number for x axis (+-1 precision)/rounding to units
t_n_ticks_y = 8  # Tick number for y axis (+-1 precision)
t_n_ticks_minor = 5  # Number of minor ticks
t_tick_length = 6  # Length of major ticks
# Grid parameters
grid_d = 'on'  # Indication of integration area for delay cut
line_type_grid_d = '-.'  # Line type for delay cut integration area
line_width_grid_d = 1.5  # Linewidth for delay cut integration area
line_op_grid_d = 100  # Opacity for delay cut integration area (0-100%)

'''Energy cut plot parameters'''
# Main parameters
line_type_e = 'o-'  # Line type for energy cut ('-','--','o-', ':', '-.')
marker_size_e = 7  # Marker size for energy cut
line_width_e = 1  # Linewidth for energy cut
line_op_e = 70  # Line opacity for energy cut (0-100%)
# Integration area indication
int_area_e = 'on'  # Indication of integration area for energy cut
line_type_int_area_e = '-'  # Line type for energy cut integration area
line_width_int_area_e = 1  # Linewidth for energy cut integration area
line_op_int_area_e = 80  # Opacity for energy cut integration area (0-100%)
# Tick parameters
e_n_ticks_x = 10  # Tick number for x axis (+-1 precision)
e_n_ticks_y = 7  # Tick number for y axis (+-1 precision)
e_n_ticks_minor = 5  # Number of minor ticks
e_tick_length = 6  # Length of major ticks
# Grid parameters
grid_e = 'on'  # Indication of integration area for delay cut
line_type_grid_e = '-.'  # Line type for delay cut integration area
line_width_grid_e = 1.5  # Linewidth for delay cut integration area
line_op_grid_e = 70  # Opacity for delay cut integration area (0-100%)
```
