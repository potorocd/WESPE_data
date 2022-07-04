# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 12:40:48 2022

@author: Dmitrii Potorochin

This program is dedicated to tr-XPS data analysis from the WESPE endstation.
"""
# This section is supposed for importing necessary modules.
import matplotlib.pyplot as plt
import numpy as np
import h5py, os, math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import matplotlib
from astropy.modeling import models, fitting
from scipy.signal import savgol_filter
from scipy import fftpack

'''
Structure of hdf5 files.

First level:
'#refs#', 'all_data', 'mess_backconv', 'param_backconvert_GUI'

For extracting data, one should go to 'all_data'.
Second level:
'DLD1Q', 'DLD4Q'
Here we should pick a specific spectrometer.

Third level:
'BAM', 'BAM_1SFELC', 'GMDBDA_Electrons', 'GMDTunnel_Electrons',
'Pulse_Energy_DiodeBB', 'angle_Grid_ROI', 'bunchID', 'delay',
'delay_corrBAM', 'delay_m', 'energy_Grid_ROI', 'file_nr',
'microbunchID', 'mono', 'time', 'time_orig', 'wip', 'x', 'xy', 'y'
Further we choose one of 1D arrays to withdray information about specific
electrons detected during the run.

'''

'''
First, we specify variables that will for the whole program.
'''
file_dir = 'D:/Data/Extension_2021_final'  # data folder
# Here we specify run numbers as a list.
run_numbers = [37378, 37379, 37380, 37381, 37382, 37383]
energy_step = 0.075  # (eV) Determines the step on the energy axis.
delay_step = 0.1  # (ps) Determines the step on the delay axis.
DLD = 'DLD4Q'  # Options: 'DLD4Q' or 'DLD1Q'
t0 = 1328.2  # For recalculation of the delay axis.
time_0 = 'on'  # Recalculation happens when time_0 = 'on'.
dif_map = 'off'  # Plotting of difference map: 'on' or 'off'. if time_0 == 'on'
t = [-0.4, 0.5]  # Here we specify delay cut positions as a list. [t1, t2, t3]
dt = [0.5, 0.5]  # Here we specify delay cut widths as a list. [dt1, dt2, dt3]
e = [202.3, 202.7, 203.2]  # Here we specify energy cut positions as a list.
de = [0.3, 0.3, 0.3]  # Here we specify energy cut widths as a list.
static = 1001  # Here we specify the del. val. where static runs will be stored

'''
Specification of ROI to zoom into

Here we can filter the data and exclude specific energy of delay regions.
Besides, there is a possibility to show spectra recorded from specific
macro - and microbunches.
'''
ROI_d = 'on'  # If 'on', the filter of delay values is activated.
ROI_e = 'on'  # If 'on', the filter of energy values is activated.
B_sorting = 'off'  # If 'on', the macrobunch filter is activated.
MB_sorting = 'off'  # If 'on', the microbunch filter is activated.
# Here we specify the delay and energy region margins as two lists.
delay_range = [-0.8, 3.5]  # delay range in ps
energy_range = [198, 206]  # kinetic energy range in eV
B_range = [0, 99]  # macrobunch range in %: [0, 50] means first 50% of bunches
MB_range = [1, 440]  # microbunch range from 1 to 440
cut_t_boundaries = 'off'  # removes the highest & the lowest delay stage values

'''
NORMALIZATION
'''
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

'''
GRAPH SETTINGS
'''
'''Graph selection'''
# Select graphs you want to see
delay_energy_map = 'on'  # Intensity versus energy and delay time both
delay_cut = 'on'  # Intensity versus energy plot (at fixed delay)
energy_cut = 'on'  # Intensity versus delay time (at fixed energy)

'''Additional features'''
# switch from 2D to 3D plot for the delay energy map
map_projection = '2D'  # '2D' or '3D' options available
elev = 80  # The elevation angle in the vertical plane in degrees.
azim = 270  # The azimuth angle in the horizontal plane in degrees.

# add difference curve to the delay cut plot
add_dif_delay_cut = 'off'  # Set 'on' if you want to get a difference plot
magn = 5  # Coefficient for multiplication of difference curves

# add vertical offset the delay cut plot (waterfall plot)
waterfall_delay_cut = 'off'  # Set 'on' to get >=0 difference between 2 curves
waterfall_offet = 0.1  # Set > 0 to increase relative shift of the curves

'''General graph parameters'''
font_family = 'Garamond'  # Figure font
font_size = 16  # Figure main font size
font_size_axis = 24  # Figure font size for axis labels
font_size_large = 30  # Figure font size for titles
dpi = 300  # Figure resolution
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

'''
FUNCTIONS
'''
# Dictionary for colors
color_dict = {
  0: 'blue',
  1: 'tab:red',
  2: 'black',
  3: 'tab:orange',
  4: 'tab:green',
  5: 'deeppink',
  6: 'tab:cyan',
  7: 'magenta',
  8: 'yellow'
}


def rounding(x, y):
    '''
    The function rounds energy and delay values to the closest values separated
    by the desired step.
    x - input value
    y - desired step
    '''
    result = (x // y) * y
    if (x / y) - (x // y) >= 0.5:
        result = result + y
    return result


def decimal_n(x):
    '''
    Determines the number of decimal points.
    '''
    result = len(str(x)) - 2
    if isinstance(x, int):
        result = 0
    return result


def non_zero_min(x):
    '''
    Finds the minimum larger than zero.
    '''
    x = np.array(x)
    filter_x = x > 0
    filter_x_arr = x[filter_x]
    result = np.min(filter_x_arr)
    return result


def non_zero_array_min(x):
    '''
    Finds the minimum larger than zero and works for arrays.
    '''
    result = []
    for i in x:
        if bool(i) and not math.isnan(i[0]) and np.mean(i) != 0:
            i = np.array(i)
            filter_i = i > 0
            filter_i_arr = i[filter_i]
            result.append(np.min(filter_i_arr))
    result = np.min(result)
    return result


def set_size(w, h, ax=None):
    """
    Helps to reduce the margins for the 3D plot.
    w, h: width, height in inches
    """
    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(right-left)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def save_dat():
    '''
    The function saves the data from Delay cut to CasaXPS-compatible format.
    '''
    for i in range(len(t)):
        x = np.expand_dims(np.array([*image_data_x]), axis=0)
        y = np.expand_dims(np.array(t_cut_plot[i]), axis=0)
        out = np.append(x, y, axis=0)
        out = np.rot90(out)
        file_full = file_dir + os.sep + 'CasaXPS' + os.sep
        file_full = file_full + f'{file_name_run}_t_{t[i]}ps_dt_{dt[i]}ps.dat'
        np.savetxt(file_full, out, delimiter='    ')


def offset_determinator(array_1, array_2, axis=0, e_offset=0, t_offset=0):
    array_1 = np.array(array_1)
    array_2 = np.array(array_2)
    if array_1.shape == array_2.shape:
        a_1 = array_1.copy()
        a_2 = array_2.copy()
        result = {}

        x, y = (0, -1)
        if e_offset != 0:
            if e_offset > 0:
                x, y = (-1, 0)
            for i in range(abs(e_offset)):
                a_2 = np.delete(a_2, x, axis=1)
                a_1 = np.delete(a_1, y, axis=1)
                # a_2 = np.insert(a_2, y, a_2[:, y], axis=1)
            array_2 = a_2.copy()
            array_1 = a_1.copy()

        x, y = (0, -1)
        if t_offset != 0:
            if t_offset > 0:
                x, y = (-1, 0)
            for i in range(abs(t_offset)):
                a_2 = np.delete(a_2, x, axis=0)
                a_1 = np.delete(a_1, y, axis=0)
                # a_2 = np.insert(a_2, y, a_2[y, :], axis=0)
            array_2 = a_2.copy()
            array_1 = a_1.copy()

        x, y = (0, -1)
        for i in [-1, 1]:
            counter = 0
            for j in range(int(np.around(array_1.shape[axis]/3,
                                         decimals=0))):
                array_dif = a_1 - a_2
                array_dif = abs(array_dif)
                array_dif = np.sum(array_dif, axis=1)
                array_dif = np.mean(array_dif)
                if axis == 0:
                    a_2 = np.delete(a_2, x, axis=0)
                    a_1 = np.delete(a_1, y, axis=0)
                    # a_2 = np.insert(a_2, y, a_2[y, :], axis=0)
                if axis == 1:
                    a_2 = np.delete(a_2, x, axis=1)
                    a_1 = np.delete(a_1, y, axis=1)
                    # a_2 = np.insert(a_2, y, a_2[:, y], axis=1)
                result[counter*i] = array_dif
                counter += 1
            x, y = (x+i, y+abs(i))
            a_1 = array_1.copy()
            a_2 = array_2.copy()

        result_sorted = {}
        for i in sorted(result.keys()):
            result_sorted[i] = result[i]
        result = result_sorted

        min_key = list(result.keys())[np.argmin(list(result.values()))]
        min_value = result[min_key]
        e_offset_eV = np.around(e_offset*energy_step,
                                decimals=decimal_n(energy_step))
        t_offset_ps = np.around(t_offset*delay_step,
                                decimals=decimal_n(delay_step))
        string_label = [f'Optimal offset: {min_key} pixels',
                        f'Minimal difference: {min_value}']
        string_title = [f'Current energy offset: {e_offset} pixels ({e_offset_eV} eV)',
                        f'Current time offset: {t_offset} pixels ({t_offset_ps} ps)']
        print(f'Optimal offset - {min_key}')
        print(f'Minimal difference - {min_value}')

        fig = plt.figure(figsize=(fig_width, fig_height*1.5), dpi=dpi)
        ax1 = fig.add_subplot(111)
        ax1.plot(result.keys(), result.values(), line_type_e,
                 color=color_dict[0],
                 label='\n'.join(string_label),
                 markersize=marker_size_e, linewidth=line_width_e)
        fig.suptitle('Offset finder', fontweight="bold",
                     fontsize=font_size_large)
        ax1.set_title('\n'.join(string_title), pad=10,
                      fontsize=font_size_axis*0.8, fontweight="light")
        ax1.set_xlabel(f'Offset along axis {axis} (pixels)', labelpad=10,
                       fontsize=font_size_axis)
        ax1.set_ylabel('Delta (arb. units)', labelpad=10,
                       fontsize=font_size_axis)
        ax1.grid(which='both', axis='both', color='lightgrey', linestyle='-.',
                 linewidth=1.1)
        ax1.axvline(min_key, color=color_dict[1])
        ax1.legend()
        plt.tight_layout()
        plt.show()
        return None
    else:
        print('Array shapes do not match!')
        return None


def t0_finder(hv=2.407, e='SB', de=0.4, function='V',
              smooth=['off', 3], der='off'):
    '''
    The function serves for t0 determination based on the first sideband
    position. It finds the kinetic energy of the most intense feature and
    makes fixed energy cut at the position separated by hv from it.
    hv: Photon energy of optical laser in eV.
    e: Energy position of the fixed energy cut.
        'SB' - cut along the first sideband
        'Main' - cut along primary feature
        int of float for user-defined value
    de: Energy window used for integration of fixed energy cut.
    function: The type of fitting function.
        'V' corresponds to Voigt.
        'G' corresponds to Gaussian.
    smooth: Data smoothing ['approach', effect]
        'approach': 'off' or 'SG' or 'FFT'
            'off': no smoothing
            'SG': Savitzky-Golay
            'FFT': fast Fourier Transform filter
        effect: int determines the power of smoothing
    der: Data derivatization (recommended for primary features)
        'on' or 'off'
    '''
    func_d = {'V': 'Voigt', 'G': 'Gaussian'}
    index_max = np.where(np.abs(image_data) == np.max(np.abs(image_data)))
    index_max_x = index_max[1][0]
    value_max_x = list(image_data_x.keys())[index_max_x]
    pos_e = value_max_x
    if e == 'SB':
        pos_e += hv
    pos_e = np.around(pos_e, 2)
    if type(e) == int or type(e) == float:
        pos_e = e

    e_bin = []
    counter = 0
    for i in image_data_x.keys():
        if i >= pos_e - de/2 and i <= pos_e + de/2:
            e_bin.append(counter)
        counter += 1

    e_cut = []
    for i in image_data:
        e_sum = 0
        for n in e_bin:
            e_sum = e_sum + i[n]
        e_cut.append(e_sum)

    if smooth[0] == 'FFT':
        e_cut_fft = fftpack.fft(e_cut)
        e_cut_power = np.abs(e_cut_fft)**2
        e_cut_freq = fftpack.fftfreq(len(e_cut), d=delay_step)
        e_cut_freq_max = np.max(e_cut_freq)
        high_freq_fft = e_cut_fft.copy()
        high_freq_fft[np.abs(e_cut_freq) > e_cut_freq_max/(smooth[1]+1)] = 0
        e_cut_filtered = fftpack.ifft(high_freq_fft)
        e_cut = e_cut_filtered
        e_cut = list(e_cut)

    if smooth[0] == 'SG' or smooth[0] == 'on':
        for i in range(smooth[1]):
            e_cut = savgol_filter(e_cut, 3, 1, mode='nearest')
        e_cut = list(e_cut)

    if der == 'on':
        e_cut = np.abs(np.gradient(e_cut))
        for i in range(3):
            e_cut = savgol_filter(e_cut, 3, 1, mode='nearest')
        e_cut = list(e_cut)

    mean_g = list(image_data_y.keys())[e_cut.index(np.max(e_cut))]
    stddev_g = len(np.where(e_cut > np.max(e_cut)/2)[0])*delay_step/2

    fitter = fitting.SLSQPLSQFitter()
    model_bg = models.Linear1D(slope=0, intercept=np.median(e_cut),
                               fixed={'slope': True},
                               bounds={'intercept': (0, np.max(e_cut)/2)})
    if function == 'G':
        model_c = models.Gaussian1D(amplitude=np.max(e_cut),
                                    mean=mean_g, stddev=stddev_g,
                                    bounds={'amplitude': (np.min(e_cut),
                                                          np.max(e_cut)),
                                            'mean':
                                                (min([*image_data_y.keys()]),
                                                 max([*image_data_y.keys()])),
                                            'stddev': (stddev_g - stddev_g/2,
                                                       stddev_g + stddev_g/2)})

    if function == 'V':
        model_c = models.Voigt1D(amplitude_L=np.max(e_cut),
                                 x_0=mean_g, fwhm_G=stddev_g, fwhm_L=stddev_g,
                                 bounds={'amplitude_L': (np.min(e_cut),
                                                         np.max(e_cut)),
                                         'x_0': (min([*image_data_y.keys()]),
                                                 max([*image_data_y.keys()])),
                                         'fwhm_G': (stddev_g - stddev_g/2,
                                                    stddev_g + stddev_g/2),
                                         'fwhm_L': (stddev_g - stddev_g/2,
                                                    stddev_g + stddev_g/2)})
    if function == 'E':
        model_c_1 = models.Exponential1D(amplitude=np.max(e_cut),
                                         tau=1,
                                         bounds={'amplitude': (np.min(e_cut),
                                                               np.max(e_cut))})
        model_c_2 = models.Exponential1D(amplitude=np.max(e_cut),
                                         tau=-1,
                                         bounds={'amplitude': (np.min(e_cut),
                                                               np.max(e_cut))})
        model_c = model_c_1 + model_c_2
        position = 0
        fwhm = 1
    if function == 'P':
        model_c = models.Polynomial1D(3, c0=1, c1=2, c2=2, c3=-2,
                                      fixed={'c0': True})
        position = 0
        fwhm = 1

    model = model_bg + model_c
    fitted_model = fitter(model, list(image_data_y.keys()), e_cut)
    if function == 'V':
        position = np.around(fitted_model.x_0_1[0], 2)
        fwhm = fitted_model.fwhm_L_1[0] + fitted_model.fwhm_G_1[0]
        fwhm = np.around(fwhm, 2)
    if function == 'G':
        position = np.around(fitted_model.mean_1[0], 2)
        fwhm = fitted_model.stddev_1[0]
        fwhm = np.around(fwhm, 2)

    if type(e) == int or type(e) == float:
        string_label_data = ['Fixed energy cut (user selected):',
                             f'E = {pos_e} eV, dE = {de} eV']
    else:
        if e == 'SB':
            string_label_data = ['Fixed energy cut across the 1$^{st}$ sideband:',
                                 f'E = {pos_e} eV, dE = {de} eV']
        else:
            string_label_data = ['Fixed energy cut across the primary feature:',
                                 f'E = {pos_e} eV, dE = {de} eV']
    string_label_fit = [f'Fit result ({func_d[function]} profile):',
                        f't$_0$ = {position} ps, FWHM = {fwhm} ps']
    if smooth == 'on' and der != 'on':
        string_label_data.append('Smoothed data')
    if der == 'on':
        string_label_data.append('Smoothed first derivative of data')

    fig = plt.figure(figsize=(fig_width, fig_height*1.25), dpi=dpi)
    ax1 = fig.add_subplot(111)
    ax1.set_title(f't$_0$ finder - {file_name_run}', pad=10,
                  fontsize=font_size_axis*1, fontweight="light")
    ax1.set_xlabel('Delay(ps)', labelpad=10,
                   fontsize=font_size_axis)
    if e_cut_norm == 'on' or map_norm == 'on' or map_norm_dif == 'on' or e_cut_norm_dif == 'on' or e_cut_norm_bin == 'on':
        ax1.set_ylabel('Intensity (arb. units)', labelpad=10,
                       fontsize=font_size_axis)
    else:
        ax1.set_ylabel('Intensity (counts)', labelpad=10,
                       fontsize=font_size_axis)
    ax1.grid(which='major', axis='both', color='lightgrey', linestyle='-.',
             linewidth=1.1)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.tick_params(axis='both', which='major', length=e_tick_length,
                    width=e_tick_length/4)
    ax1.tick_params(axis='both', which='minor', length=e_tick_length/1.5,
                    width=e_tick_length/4)
    ax1.plot(image_data_y.keys(), e_cut, 'o',
             color=color_dict[0],
             label='\n'.join(string_label_data),
             markersize=marker_size_e, linewidth=line_width_e)

    ax1.plot(image_data_y.keys(), fitted_model(list(image_data_y.keys())),
             line_type_e, color=color_dict[1],
             label='\n'.join(string_label_fit),
             markersize=marker_size_e, linewidth=line_width_e)
    ax1.axvline(position, color=color_dict[2])
    ax1.legend()
    plt.tight_layout()
    plt.show()
    print('\n'.join(string_label_fit))
    print(fitted_model)
    return position


def run_corrector(run_number, e_shift=0, t_shift=0):
    """
    This function serves to shift the whole run in the energy and the time
    domain. One should specify three values: the run number, the two values
    added to the energy, and the delay axes. The Matlab file will be
    permanently modified; save the original if you want to have a chance
    to recover it.
    run_number : run number for the correction
    e_shift : the value added to the energy axis
    t_shift : the value added to the delay axis (to the original delay
                                                 stage values)
    """
    file_name = f'{run_number}' + os.sep + f'{run_number}_energy.mat'
    file_full = file_dir + os.sep + file_name
    f = h5py.File(file_full, 'r+')
    for DLD in ['DLD1Q', 'DLD4Q']:
        DLD_energy = f.get(f'all_data/{DLD}/energy_Grid_ROI')[0]
        DLD_energy += e_shift
        DLD_energy = np.expand_dims(DLD_energy, axis=0)
        del f[f'all_data/{DLD}/energy_Grid_ROI']
        f.create_dataset(f'all_data/{DLD}/energy_Grid_ROI',
                         data=DLD_energy)
        DLD_delay = f.get(f'all_data/{DLD}/delay')[0]
        DLD_delay += t_shift
        DLD_delay = np.expand_dims(DLD_delay, axis=0)
        del f[f'all_data/{DLD}/delay']
        f.create_dataset(f'all_data/{DLD}/delay',
                         data=DLD_delay)
    f.close()
    return None


def run_corrector_to_static(run_number):
    """
    This function removes delay stage values from a Matlab file.
    This action is required when we scan the delay stage when the laser is off.
    run_number : run number for the correction
    """
    file_name = f'{run_number}' + os.sep + f'{run_number}_energy.mat'
    file_full = file_dir + os.sep + file_name
    f = h5py.File(file_full, 'r+')
    for DLD in ['DLD1Q', 'DLD4Q']:
        del f[f'all_data/{DLD}/delay']
    f.close()
    return None


def is_static(run_number):
    """
    The function checks if delay stage values are stored in a Matlab file.
    If there are no delay values, it returns True, i.e.,
    the run is considered static.
    run_number : run number for the correction
    """
    file_name = f'{run_number}' + os.sep + f'{run_number}_energy.mat'
    file_full = file_dir + os.sep + file_name
    f = h5py.File(file_full, 'r')
    try:
        for DLD in ['DLD1Q', 'DLD4Q']:
            f[f'all_data/{DLD}/delay']
    except KeyError:
        return True
    return False
    f.close()


def cur_dir():
    os.chdir(os.path.abspath(os.path.dirname(__file__)))


'''
Several tricks to avoid bugs when wrong values are entered.
'''
t0 = rounding(t0, delay_step)
try:
    if time_0 == 'on' and max(t) > 1000:
        for i in range(len(t)):
            t[i] = np.around(t0 - t[i], decimals=decimal_n(delay_step))

    if time_0 != 'on' and max(t) < 1000:
        for i in range(len(t)):
            t[i] = np.around(t0 - t[i], decimals=decimal_n(delay_step))

    if ROI_d == 'on' and max(delay_range) < 1000:
        for i in range(len(delay_range)):
            delay_range[i] = np.around(t0 - delay_range[i],
                                       decimals=decimal_n(delay_step))
except ValueError:
    pass


'''In case one forgets to specify dt or de'''

if len(dt) < len(t):  # filling in missing dt values
    for i in range(len(t)):
        try:
            dt[i]
        except IndexError:
            try:
                dt.append(dt[i-1])
            except IndexError:
                dt.append(0.5)

if len(de) < len(e):  # filling in missing de values
    for i in range(len(e)):
        try:
            de[i]
        except IndexError:
            try:
                de.append(de[i-1])
            except IndexError:
                de.append(0.2)

'''Figure parameters determination based on the number of subplots'''

fig_number = 3
fig_pos_1 = 1
fig_pos_2 = 2
fig_pos_3 = 3

if delay_energy_map == 'off':
    fig_number -= 1
    fig_pos_1 -= 1
    fig_pos_2 -= 1
    fig_pos_3 -= 1

if delay_cut == 'off':
    fig_number -= 1
    fig_pos_2 -= 1
    fig_pos_3 -= 1

if energy_cut == 'off':
    fig_number -= 1
    fig_pos_3 -= 1


'''
Data manipulation starts here.
We upload the data from '.mat' files.
'''
for i in run_numbers:
    if i == run_numbers[0]:  # excecuted for the first scan from the list
        file_name = f'{i}' + os.sep + f'{i}_energy.mat'
        file_full = file_dir + os.sep + file_name
        f = h5py.File(file_full, 'r')
        DLD_energy = f.get(f'all_data/{DLD}/energy_Grid_ROI')[0]
        B_ID = f.get(f'all_data/{DLD}/bunchID')[0]
        MB_ID = f.get(f'all_data/{DLD}/microbunchID')[0]
        try:
            DLD_delay = f.get(f'all_data/{DLD}/delay')[0]
        except TypeError:
            DLD_delay = np.full(DLD_energy.shape, static)
            run_numbers[run_numbers.index(i)] = f'{i}_static'
            if time_0 == 'on':
                t.insert(0, np.around(t0-static, decimal_n(delay_step)))
            else:
                t.insert(0, static)
            dt.insert(0, 0.5)
            static += 1
    else:  # excecuted for further scans
        file_name = f'{i}' + os.sep + f'{i}_energy.mat'
        file_full = file_dir + os.sep + file_name
        f = h5py.File(file_full, 'r')
        DLD_energy_i = f.get(f'all_data/{DLD}/energy_Grid_ROI')[0]
        B_ID_i = f.get(f'all_data/{DLD}/bunchID')[0]
        MB_ID_i = f.get(f'all_data/{DLD}/microbunchID')[0]
        try:  # check if we have a static scan
            DLD_delay_i = f.get(f'all_data/{DLD}/delay')[0]
        except TypeError:  # if static - we make up missing delay values
            DLD_delay_i = np.full(DLD_energy_i.shape, static)
            run_numbers[run_numbers.index(i)] = f'{i}_static'
            if time_0 == 'on':
                t.insert(0, np.around(t0-static, decimal_n(delay_step)))
            else:
                t.insert(0, static)
            dt.insert(0, 0.5)
            static += 1
        #  here we add up everything to one array
        DLD_delay = np.append(DLD_delay, DLD_delay_i)
        DLD_energy = np.append(DLD_energy, DLD_energy_i)
        B_ID = np.append(B_ID, B_ID_i)
        MB_ID = np.append(MB_ID, MB_ID_i)
# rounding of the arrays
DLD_delay_r = [rounding(i, delay_step) for i in DLD_delay]
DLD_energy_r = [rounding(i, energy_step) for i in DLD_energy]
DLD_delay_r = np.around(DLD_delay_r, decimals=decimal_n(delay_step))
DLD_energy_r = np.around(DLD_energy_r, decimals=decimal_n(energy_step))
f.close()

'''
Here we create a dictionary (delay_info), which stores all
delay values as keys.
Then, we assign delays to the numbers of count events.
'''
counter = 0
delay_info = {}
for i in DLD_delay_r:
    if i > 0:
        if ROI_d == 'on':
            if i >= min(delay_range) and i <= max(delay_range):
                if i not in delay_info.keys():
                    delay_info[i] = [counter]
                else:
                    delay_info[i].append(counter)
        else:
            if i not in delay_info.keys():
                delay_info[i] = [counter]
            else:
                delay_info[i].append(counter)
    counter += 1

'''
Further, we sort the dictionary in ascending order of
key values (delay values).
'''
delay_info_sorted = {}
for i in sorted(delay_info.keys()):
    delay_info_sorted[i] = delay_info[i]


delay_info_sorted_t0 = {}
if time_0 == 'on':
    for i in delay_info_sorted.keys():
        delay_info_sorted_t0[np.around(t0-i,
                      decimals=decimal_n(delay_step))] = delay_info_sorted[i]
    delay_info_sorted = delay_info_sorted_t0


'''Bunch sorting'''

B_min = np.min(B_ID)+(np.max(B_ID)-np.min(B_ID))*min(B_range)/100
B_max = np.min(B_ID)+(np.max(B_ID)-np.min(B_ID))*max(B_range)/100

delay_info_sorted_B = {}
if B_sorting == 'on':
    for i in delay_info_sorted.keys():
        B_sorted_list = []
        for j in delay_info_sorted[i]:
            if B_ID[j] >= B_min and B_ID[j] <= B_max:
                B_sorted_list.append(j)
        delay_info_sorted_B[i] = B_sorted_list
    delay_info_sorted = delay_info_sorted_B

'''Microbunch sorting'''

delay_info_sorted_MB = {}
if MB_sorting == 'on':
    for i in delay_info_sorted.keys():
        MB_sorted_list = []
        for j in delay_info_sorted[i]:
            if MB_ID[j] >= min(MB_range) and MB_ID[j] <= max(MB_range):
                MB_sorted_list.append(j)
        delay_info_sorted_MB[i] = MB_sorted_list
    delay_info_sorted = delay_info_sorted_MB

'''
Here we create the main database for the whole file:
    delay_energy_data[i][j][k]
    i is responsible for the time axis, within every element we have three
    cells, [j = 0, 1, 2] the first cell stores delay value, the second cell
    stores a list containing kinetic energies, the third cell contains the
    number of counts detected.
'''
delay_energy_data = []
for i in delay_info_sorted.keys():  # We go through every delay one by one.
    energy_info = {}
    for event in delay_info_sorted[i]:  # We go through every count one by one.
        if DLD_energy_r[event] not in energy_info.keys():
            energy_info[DLD_energy_r[event]] = 1
        else:
            energy_info[DLD_energy_r[event]] += 1  # Counting of electrons.
    for eng in np.arange(min(DLD_energy_r), max(DLD_energy_r)
                         + energy_step, energy_step):  # when counts[E] = 0
        if np.around(eng, decimals=
                     decimal_n(energy_step)) not in energy_info.keys():
            energy_info[np.around(eng, decimals=decimal_n(energy_step))] = 0
    energy_list = sorted(energy_info.keys())  # Sorting of KE
    if ROI_e == 'on':
        energy_list = [i for i in energy_list if i >= min(energy_range)
                       and i <= max(energy_range)]
    intensity_list = []
    for energy in energy_list:  # Sorting of Counts along KE
        intensity_list.append(energy_info[energy])
    delay_energy_data.append([i, energy_list, intensity_list])
    # Every cycle creates a cell [delay, [KE], [Counts]]

'''

CREATING ENERGY-DELAY MAP

image_data - the map itself
image_data_x - dictionary, where keys correspond to energy values,
values correspond to the number of pixels along the x-axis
image_data_y - dictionary, where keys() correspond to delay,
values() correspond to the number of pixels along the y-axis

'''
image_data = []
for i in delay_energy_data:
    if map_norm == 'on':
        if min(i[2]) < 0:
            y = [(k - min(i[2])) for k in i[2]]
        y = [(k - min(i[2])) for k in i[2]]
        y = [k / max(y) for k in y]
        image_data.append(y)
    else:
        image_data.append(i[2])

if total_electron_norm == 'on':
    max_sum = np.max(np.sum(image_data, axis=1))
    image_data_norm = []
    for i in image_data:
        line = [k*max_sum/sum(i) for k in i]
        image_data_norm.append(line)
    image_data = image_data_norm

image_data_x = {}
counter = 0
for i in delay_energy_data[0][1]:
    image_data_x[i] = counter
    counter += 1

image_data_y_i = []
for i in delay_energy_data:
    image_data_y_i.append(i[0])

image_data_y = {}
counter = 0
for i in image_data_y_i:
    image_data_y[i] = counter
    counter += 1

if cut_t_boundaries == 'on':
    image_data = image_data[1:-1]
    image_data_y.pop(list(image_data_y.keys())[0])
    image_data_y.pop(list(image_data_y.keys())[-1])

if min(np.array(image_data).shape) == 0:
    raise ValueError('Check ROI values!\n'
                     'Probably,the request does not fit the '
                     'delay-energy data.')

'''
MAKING DIFFERENCE MAP
'''
if time_0 == 'on':
    if dif_map == 'on':
        image_data_dif = []
        t_bin = []
        counter = 0
        for i in image_data_y.keys():
            if i < -0.2 and i > min([*image_data_y.keys()]):
                t_bin.append(counter)
            counter += 1

        norm_cut = []
        for i in t_bin:
            norm_cut.append(image_data[i])
        norm_cut = np.mean(norm_cut, axis=0)
        for i in image_data:
            image_data_dif.append(np.subtract(i, norm_cut))
        image_data = image_data_dif


if map_norm_dif == 'on':
    image_data_dif_norm = []
    pos_norm = abs(np.max(image_data))
    neg_norm = abs(np.min(image_data))
    norm = max(neg_norm, pos_norm)
    for i in image_data:
        line = []
        for j in i:
            element = j/norm
            line.append(element)
        image_data_dif_norm.append(line)
    image_data = image_data_dif_norm

'''
MAKING DELAY CUTS
'''
if delay_cut == 'on':
    t_cut_plot = []
    for j in range(len(t)):
        t_bin = []
        counter = 0
        for i in image_data_y.keys():
            if i >= t[j] - dt[j]/2 and i <= t[j] + dt[j]/2:
                t_bin.append(counter)
            counter += 1

        t_cut = []
        for i in t_bin:
            if i == t_bin[0]:
                t_cut = image_data[i]
            else:
                t_cut = [a + b for a, b in zip(t_cut, image_data[i])]

        if t_cut_norm_bin == 'on':
            t_cut = [k/len(t_bin) for k in t_cut]

        if t_cut_norm == 'on':
            try:
                if min(t_cut) < 0:
                    t_cut = [(k - min(t_cut)) for k in t_cut]
            except ValueError:
                pass
            t_cut = [(k - min(t_cut)) for k in t_cut]
            t_cut = [k / max(t_cut) for k in t_cut]
        t_cut_plot.append(t_cut)

    if t_cut_norm_dif == 'on':
        t_cut_plot_dif = []
        pos_norm = abs(np.max(t_cut_plot))
        neg_norm = abs(np.min(t_cut_plot))
        norm = max(neg_norm, pos_norm)
        for i in t_cut_plot:
            line = []
            for j in i:
                element = j/norm
                line.append(element)
            t_cut_plot_dif.append(line)
        t_cut_plot = t_cut_plot_dif

    if add_dif_delay_cut == 'on':
        counter = 0
        t_cut_plot_dif = t_cut_plot
        for i in range(len(t_cut_plot)):
            if counter == 0:
                reference = t_cut_plot[i]
            else:
                dif_line = [a - b for a, b in zip(t_cut_plot[i], reference)]
                if magn > 1 or magn < 1:
                    dif_line = [a*magn for a in dif_line]
                    t.append(f'Difference T$_{counter+1}$-T$_1$ x {magn}')
                else:
                    t.append(f'Difference T$_{counter+1}$-T$_1$')
                t_cut_plot_dif.append(dif_line)
                dt.append('dT')
            counter += 1
        t_cut_plot = t_cut_plot_dif

    if waterfall_delay_cut == 'on':
        for i in range(len(t_cut_plot)-1):
            t_cut_plot_wf_1 = np.delete(np.array(t_cut_plot), -1, axis=0)
            t_cut_plot_wf_2 = np.delete(np.array(t_cut_plot), 0, axis=0)
            t_cut_plot_wf_delta = t_cut_plot_wf_2 - t_cut_plot_wf_1
            t_cut_plot_wf_delta = np.abs(np.min(t_cut_plot_wf_delta, axis=1))
            t_cut_plot_wf_delta = list(t_cut_plot_wf_delta)
            t_cut_plot_wf = []
            counter = 0
            for i in t_cut_plot:
                if counter == 0:
                    t_cut_plot_wf.append(i)
                else:
                    line = [a + t_cut_plot_wf_delta[counter - 1] for a in i]
                    t_cut_plot_wf.append(line)
                counter += 1
            t_cut_plot = t_cut_plot_wf

        t_cut_plot_wf = []
        if waterfall_offet > 0:
            counter = 0
            for i in t_cut_plot:
                if counter == 0:
                    t_cut_plot_wf.append(i)
                else:
                    line = [a + waterfall_offet*counter for a in i]
                    t_cut_plot_wf.append(line)
                counter += 1
            t_cut_plot = t_cut_plot_wf

    if min(np.array(t_cut_plot).shape) == 0:
        raise ValueError('Check values for the delay cut plot!\n'
                         'Probably,the request does not fit the '
                         'delay-energy data.')

'''
MAKING ENERGY CUTS
'''
if energy_cut == 'on':
    e_cut_plot = []
    for j in range(len(e)):
        e_bin = []
        counter = 0
        for i in image_data_x.keys():
            if i >= e[j] - de[j]/2 and i <= e[j] + de[j]/2:
                e_bin.append(counter)
            counter += 1

        e_cut = []
        for i in image_data:
            e_sum = 0
            for n in e_bin:
                e_sum = e_sum + i[n]
            e_cut.append(e_sum)

        if e_cut_norm_bin == 'on':
            e_cut = [k/len(e_bin) for k in e_cut]

        if e_cut_norm == 'on':
            if min(t_cut) < 0:
                t_cut = [(k - min(t_cut)) for k in t_cut]
            e_cut = [(k - min(e_cut)) for k in e_cut]
            e_cut = [k / np.mean(
                e_cut[len(e_cut)-6:len(e_cut)-1]) for k in e_cut]
        e_cut_plot.append(e_cut)

    if e_cut_norm_dif == 'on':
        e_cut_plot_dif = []
        pos_norm = abs(np.max(e_cut_plot))
        neg_norm = abs(np.min(e_cut_plot))
        norm = max(neg_norm, pos_norm)
        for i in e_cut_plot:
            line = []
            for j in i:
                element = j/norm
                line.append(element)
            e_cut_plot_dif.append(line)
        e_cut_plot = e_cut_plot_dif

    if len(e_cut_plot) == 0:
        for i in image_data_y.keys():
            e_cut_plot.append([1])

    if min(np.array(e_cut_plot).shape) == 0:
        raise ValueError('Check values for the energy cut plot!\n'
                         'Probably,the request does not fit the '
                         'delay-energy data.')

'''

FIGURE RENDERING WITH MATPLOTLIB

'''
'''Tick parameters determination (Delay-energy map)'''
if delay_energy_map == 'on':
    map_z_max = np.nanmax(image_data)
    map_z_min = np.nanmin(image_data)
    map_z_tick = (map_z_max - map_z_min)/map_n_ticks_z
    if map_z_tick < 1:
        map_z_tick_decimal = 1
    else:
        map_z_tick_decimal = 0

    map_y_max = np.nanmax(np.nanmax(list(image_data_y.keys())))
    map_y_min = np.nanmax(np.nanmin(list(image_data_y.keys())))
    map_y_tick = (map_y_max - map_y_min)/map_n_ticks_y
    if map_y_tick < 1:
        map_y_tick_decimal = 1
    else:
        map_y_tick_decimal = 0

    map_x_max = np.nanmax(np.nanmax(list(image_data_x.keys())))
    map_x_min = np.nanmin(np.nanmin(list(image_data_x.keys())))
    map_x_tick = (map_x_max - map_x_min)/map_n_ticks_x
    map_x_tick_decimal = 0

'''Tick parameters determination (Delay cut)'''

if delay_cut == 'on':
    t_y_max = np.nanmax(np.nanmax(np.array(t_cut_plot)))
    t_y_min = non_zero_array_min(t_cut_plot)
    if dif_map == 'on' or add_dif_delay_cut == 'on':
        t_y_min = np.nanmin(np.nanmin(np.array(t_cut_plot)))
    t_y_tick = (t_y_max - t_y_min)/t_n_ticks_y
    t_y_tick = (t_y_max - t_y_min + t_y_tick)/t_n_ticks_y
    if t_y_tick < 1:
        t_y_tick_decimal = 1
    else:
        t_y_tick_decimal = 0

    t_x_max = np.nanmax(np.nanmax(list(image_data_x.keys())))
    t_x_min = np.nanmin(np.nanmin(list(image_data_x.keys())))
    t_x_tick = (t_x_max - t_x_min)/t_n_ticks_x
    t_x_tick_decimal = 0

'''Tick parameters determination (Energy cut)'''

if energy_cut == 'on':
    e_y_max = np.nanmax(np.nanmax(np.array(e_cut_plot)))
    e_y_min = non_zero_array_min(e_cut_plot)
    if dif_map == 'on':
        e_y_min = np.nanmin(np.nanmin(np.array(e_cut_plot)))
    e_y_tick = (e_y_max - e_y_min)/e_n_ticks_y
    e_y_tick = (e_y_max - e_y_min + e_y_tick*2)/e_n_ticks_y
    if e_y_tick < 1:
        e_y_tick_decimal = 1
    else:
        e_y_tick_decimal = 0

    e_x_max = np.nanmax(np.nanmax(list(image_data_y.keys())))
    e_x_min = np.nanmin(np.nanmin(list(image_data_y.keys())))
    e_x_tick = (e_x_max - e_x_min)/e_n_ticks_x
    if e_x_tick < 1:
        e_x_tick_decimal = 1
    else:
        e_x_tick_decimal = 0

    if e_y_tick == 0:
        e_y_tick = 1

    if e_x_tick == 0:
        e_x_tick = 1

'''General graph parameters (font, dpi, size)'''

matplotlib.rcParams.update({'font.size': font_size,
                            'font.family': font_family,
                            'axes.linewidth': axes_linewidth})  # Setting fonts

fig = plt.figure(figsize=(fig_width, fig_height*fig_number), dpi=dpi)

'''
FORMING THE MAIN TITLE
'''

if len(run_numbers) == 1:
    fig.suptitle(f'Run {run_numbers[0]}', fontweight="bold",
                 fontsize=font_size_large)
    file_name_run = f'Run_{run_numbers[0]}'
elif len(run_numbers) > 4:
    fig.suptitle(f'Runs {run_numbers[0]}-{run_numbers[-1]}', fontweight="bold",
                 fontsize=font_size_large)
    file_name_run = f'Runs_{run_numbers[0]}-{run_numbers[-1]}'
else:
    run_string = str(run_numbers[0])
    for i in run_numbers[1:]:
        run_string = run_string + ' & ' + str(i)
    fig.suptitle(f'Runs {run_string}', fontweight="bold",
                 fontsize=font_size_large)
    file_name_run = f'Runs {run_string}'

'''
PLOTTING DELAY SCAN MAP
'''
if delay_energy_map == 'on':
    if dif_map == 'on' and map_norm_dif == 'on':
        vmin = -1
        vmax = 1
    elif dif_map == 'on':
        vmin = np.min(image_data)
        vmax = np.max(image_data)
    else:
        vmin = 0.05*np.max(image_data)
        vmax = 0.95*np.max(image_data)

    if map_projection == '3D':
        ax1 = fig.add_subplot(fig_number, 1, fig_pos_1, projection='3d')
        x = np.array([*image_data_x])
        y = np.array([*image_data_y])
        X, Y = np.meshgrid(x, y)
        Z = np.array(image_data)
        im1 = ax1.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0,
                               antialiased=False, vmin=vmin, vmax=vmax,
                               rstride=1, cstride=1)

        ax1.view_init(elev=elev, azim=azim)
        ax1.set_box_aspect(aspect=(1, 1, 1))
        set_size(fig_width, fig_height/0.8, ax=ax1)
        ax1.set_title('Delay scan map', pad=0, fontsize=font_size_axis*1.2,
                      fontweight="light")
        ax1.set_xlabel('Kinetic energy (eV)', labelpad=20,
                       fontsize=font_size_axis)
        ax1.set_ylabel('Delay (ps)', labelpad=15, fontsize=font_size_axis*0.8)
        ax1.zaxis.set_major_locator(matplotlib.ticker.LinearLocator(0))

        cbar = fig.colorbar(im1,
                            ticks=MultipleLocator(round(map_z_tick,
                                                        map_z_tick_decimal)))
        if map_norm == 'on':
            cbar.set_label('Intensity (arb. units)', rotation=270, labelpad=30,
                           fontsize=font_size_axis*0.8)
        else:
            cbar.set_label('Intensity (counts)', rotation=270, labelpad=30,
                           fontsize=font_size_axis*0.8)

    if map_projection == '2D':
        ax1 = fig.add_subplot(fig_number, 1, fig_pos_1)
        if time_0 == 'on':
            extent = [min(image_data_x), max(image_data_x),
                      min(image_data_y), max(image_data_y)]
        else:
            extent = [min(image_data_x), max(image_data_x),
                      max(image_data_y), min(image_data_y)]

        im1 = ax1.imshow(image_data, origin='upper',
                         extent=extent,
                         vmin=vmin,
                         vmax=vmax,
                         cmap="coolwarm", aspect='auto')

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="3.5%", pad=0.09)
        cbar = fig.colorbar(im1, cax=cax1,
                            ticks=MultipleLocator(round(map_z_tick,
                                                        map_z_tick_decimal)))
        cbar.minorticks_on()
        if map_norm == 'on' or map_norm_dif == 'on':
            cax1.set_ylabel('Intensity (arb. units)', rotation=270,
                            labelpad=20, fontsize=font_size_axis*0.8)
        else:
            cax1.set_ylabel('Intensity (counts)', rotation=270, labelpad=20,
                            fontsize=font_size_axis*0.8)

        ax1.set_title('Delay scan map', pad=15, fontsize=font_size_axis*1.2,
                      fontweight="light")
        ax1.set_xlabel('Kinetic energy (eV)', labelpad=10,
                       fontsize=font_size_axis)
        ax1.set_ylabel('Delay (ps)', labelpad=10, fontsize=font_size_axis*0.8)
        if t0_line == 'on':
            ax1.axhline(y=0, color=color_t0_line,
                        linewidth=line_width_t0_line,
                        alpha=line_op_t0_line/100,
                        linestyle=line_type_t0_line)

    # y axis
    ax1.yaxis.set_major_locator(MultipleLocator(round(map_y_tick,
                                                      map_y_tick_decimal)))
    ax1.yaxis.set_minor_locator(MultipleLocator(round(map_y_tick,
                                                      map_y_tick_decimal) /
                                                map_n_ticks_minor))
    # x axis
    ax1.xaxis.set_major_locator(MultipleLocator(round(map_x_tick,
                                                      map_x_tick_decimal)))
    ax1.xaxis.set_minor_locator(MultipleLocator(round(map_x_tick,
                                                      map_x_tick_decimal) /
                                                map_n_ticks_minor))
    ax1.tick_params(axis='both', which='major', length=map_tick_length,
                    width=map_tick_length/4)
    ax1.tick_params(axis='both', which='minor', length=map_tick_length/1.5,
                    width=map_tick_length/4)
    cax1.tick_params(axis='both', which='major', length=map_tick_length,
                     width=map_tick_length/4)
    cax1.tick_params(axis='both', which='minor', length=map_tick_length/1.5,
                     width=map_tick_length/4)
    ax1.set_ylim(map_y_min, map_y_max)
    ax1.set_xlim(map_x_min, map_x_max)

'''
PLOTTING DELAY CUTS
'''
if delay_cut == 'on':
    ax2 = fig.add_subplot(fig_number, 1, fig_pos_2)
    if not math.isnan(np.mean(np.mean(np.array(t_cut_plot, dtype='object')))):
        ax2.set_title('Delay cut comparison', pad=15,
                      fontsize=font_size_axis*1.2,
                      fontweight="light")
        for i in range(len(t)):
            if bool(t_cut_plot[i]) and not math.isnan(t_cut_plot[i][0]):
                if type(t[i]) == str:
                    label = t[i]
                else:
                    label = f'T$_{i+1}$ = {t[i]} ps, dT$_{i+1}$ = {dt[i]} ps'
                ax2.plot(image_data_x.keys(), t_cut_plot[i], line_type_d,
                         color=color_dict[i],
                         label=label,
                         markersize=marker_size_d, linewidth=line_width_d,
                         alpha=line_op_d/100)
                if int_area_d == 'on' and delay_energy_map == 'on':
                    try:
                        ax1.axhline(y=t[i]+dt[i]/2, color=color_dict[i],
                                    linewidth=line_width_int_area_d,
                                    alpha=line_op_int_area_d/100,
                                    linestyle=line_type_int_area_d)
                        ax1.axhline(y=t[i]-dt[i]/2, color=color_dict[i],
                                    linewidth=line_width_int_area_d,
                                    alpha=line_op_int_area_d/100,
                                    linestyle=line_type_int_area_d)
                    except TypeError:
                        pass
        ax2.set_xlabel('Kinetic energy (eV)', labelpad=10,
                       fontsize=font_size_axis)

        if t_cut_norm == 'on' or map_norm == 'on' or map_norm_dif == 'on' or t_cut_norm_dif == 'on' or t_cut_norm_bin == 'on':
            ax2.set_ylabel('Intensity (arb. units)', labelpad=10,
                           fontsize=font_size_axis)
        else:
            ax2.set_ylabel('Intensity (counts)', labelpad=10,
                           fontsize=font_size_axis)
        # y axis
        ax2.yaxis.set_major_locator(MultipleLocator(round(t_y_tick,
                                                          t_y_tick_decimal)))
        ax2.set_ylim(t_y_min-t_y_tick/2, t_y_max + t_y_tick)
        ax2.yaxis.set_minor_locator(MultipleLocator(round(t_y_tick,
                                                          t_y_tick_decimal) /
                                                    t_n_ticks_minor))
        # x axis
        ax2.xaxis.set_major_locator(MultipleLocator(round(t_x_tick,
                                                          t_x_tick_decimal)))
        ax2.xaxis.set_minor_locator(MultipleLocator(round(t_x_tick,
                                                          t_x_tick_decimal) /
                                                    t_n_ticks_minor))
        ax2.set_xlim(t_x_min, t_x_max)
        ax2.legend()
        if grid_d == 'on':
            ax2.grid(which='major', axis='both', color='lightgrey',
                     linestyle=line_type_grid_d, alpha=line_op_grid_d/100,
                     linewidth=line_width_grid_d)
        ax2.tick_params(axis='both', which='major', length=t_tick_length,
                        width=t_tick_length/4)
        ax2.tick_params(axis='both', which='minor', length=t_tick_length/1.5,
                        width=t_tick_length/4)

'''
PLOTTING ENERGY CUTS
'''
if energy_cut == 'on':
    ax3 = fig.add_subplot(fig_number, 1, fig_pos_3)
    if not math.isnan(np.mean(np.mean(np.array(e_cut_plot, dtype='object')))):
        ax3.set_title('Energy cut comparison', pad=15,
                      fontsize=font_size_axis*1.2,
                      fontweight="light")
        for i in range(len(e)):
            if bool(e_cut_plot[i]) and not math.isnan(e_cut_plot[i][0]):
                ax3.plot(image_data_y.keys(), e_cut_plot[i], line_type_e,
                         color=color_dict[i],
                         label=f'E$_{i+1}$ = {e[i]} eV, dE$_{i+1}$ = {de[i]} eV',
                         markersize=marker_size_e, linewidth=line_width_e,
                         alpha=line_op_e/100)
                if int_area_e == 'on':
                    ax1.axvline(x=e[i]+de[i]/2, color=color_dict[i],
                                linewidth=line_width_int_area_e,
                                alpha=line_op_int_area_e/100,
                                linestyle=line_type_int_area_e)
                    ax1.axvline(x=e[i]-de[i]/2, color=color_dict[i],
                                linewidth=line_width_int_area_e,
                                alpha=line_op_int_area_e/100,
                                linestyle=line_type_int_area_e)
        ax3.set_xlabel('Delay (ps)', labelpad=10, fontsize=font_size_axis)
        if e_cut_norm == 'on' or map_norm == 'on' or map_norm_dif == 'on' or e_cut_norm_dif == 'on' or e_cut_norm_bin == 'on':
            ax3.set_ylabel('Intensity (arb. units)', labelpad=10,
                           fontsize=font_size_axis)
        else:
            ax3.set_ylabel('Intensity (counts)', labelpad=10,
                           fontsize=font_size_axis)
        # y axis
        ax3.yaxis.set_major_locator(MultipleLocator(round(e_y_tick,
                                                          e_y_tick_decimal)))
        ax3.set_ylim(e_y_min-e_y_tick, e_y_max + e_y_tick)
        ax3.yaxis.set_minor_locator(MultipleLocator(round(e_y_tick,
                                                          e_y_tick_decimal) /
                                                    e_n_ticks_minor))
        # x axis
        ax3.xaxis.set_major_locator(MultipleLocator(round(e_x_tick,
                                                          e_x_tick_decimal)))
        ax3.xaxis.set_minor_locator(MultipleLocator(round(e_x_tick,
                                                          e_x_tick_decimal) /
                                                    e_n_ticks_minor))
        ax3.set_xlim(e_x_min, e_x_max)
        ax3.legend()
        if grid_e == 'on':
            ax3.grid(which='major', axis='both', color='lightgrey',
                     linestyle=line_type_grid_e, alpha=line_op_grid_e/100,
                     linewidth=line_width_grid_e)
        if time_0 != 'on':
            ax3.invert_xaxis()
        ax3.tick_params(axis='both', which='major', length=e_tick_length,
                        width=e_tick_length/4)
        ax3.tick_params(axis='both', which='minor', length=e_tick_length/1.5,
                        width=e_tick_length/4)


plt.subplots_adjust(top=0.95)
if ROI_e == 'on':
    plt.subplots_adjust(top=0.92)
plt.tight_layout()
plt.show()
