'''
TODO: double check that mzi_peak_freqs plot gets removed at some point
'''
import pandas as pd
import numpy as np
from pathlib import Path
import pyqtgraph as pg
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from PyQt6.QtWidgets import QWidget, QSplitter
from PyQt6.QtGui import QPen, QColor
from pyqtgraph.parametertree import (
    Parameter, 
    ParameterTree, 
    RunOptions,
    Interactor,
    interact
)

app = pg.mkQApp('Test')
plot_widget = pg.GraphicsLayoutWidget()
plot_widget.ci.setBorder((50, 50, 100))

win = QSplitter()
win.resize(1000, 500)

ptree = ParameterTree(showHeader=False)
params = Parameter.create(name='Test', type='group')
file_in = Parameter.create(name='Choose File', type='file')
ptree.addParameters(file_in)
ptree.addParameters(params)
ptree.resize(400,500)

interactor = Interactor(parent=params, runOptions=RunOptions.ON_ACTION)

win.addWidget(plot_widget)
win.addWidget(ptree)
win.show()

norm = lambda arr: (arr - np.min(arr)) / np.ptp(arr)
hz_to_samples = lambda fsr: round( fsr * abs( ( sample_rate * start_wl ** 2 ) / ( c * velocity ) ) )

mzi_data = []
transmission_data = []
time = []

sample_rate = 0
velocity = 0
start_wl = 0
mzi_bw = 16e6
res_fsr = 0
# velocity = -100e-9
# start_wl = 1638e-9
# res_fsr = 15e9
c = 299792458

resonance_info = pd.DataFrame()

transmission_peaks_freq = []
transmission_peaks_Q = []
mzi_peak_times = []
mzi_peak_freq = []

def load_data(path, mzi_smoothing_sigma=5):
    global mzi_data, transmission_data, time, sample_rate
    data_mat = scipy.io.loadmat(path)

    mzi = data_mat['mzi'][0]
    good_data = ~np.isinf(mzi)

    mzi_data = norm(gaussian_filter1d(mzi[good_data], mzi_smoothing_sigma))

    transmission_data = data_mat['transmission'][0][good_data]
    transmission_data /= np.max(transmission_data)
#    print(np.min(transmission_data))
    time = data_mat['Tinterval'][0][0] * np.arange(len(mzi))[good_data] + data_mat['Tstart'][0][0]
    sample_rate = len(time) / np.ptp(time)

    for plot_item in plot_items:
        plot_item.clear()

    plot_items[0].addItem(
        pg.PlotDataItem(x=time, y=mzi_data, name='mzi')
    )
    plot_items[1].addItem(
        pg.PlotDataItem(x=time, y=transmission_data, name='transmission')
    )

def initialize(laser_velocity_nms=1e9*velocity, start_wavelength_nm=1e9*start_wl, mzi_bandwidth_hz=mzi_bw, mzi_smoothing_sigma=5):
    global velocity, start_wl, mzi_bw
    assert file_in.value() is not None, 'Must select input file to initialize'
    velocity = laser_velocity_nms * 1e-9
    start_wl = start_wavelength_nm * 1e-9
    mzi_bw = mzi_bandwidth_hz
    load_data(file_in.value(), mzi_smoothing_sigma)
initialization_params = interact(initialize, parent=params)

def crop(time_start=0.0, time_end=0.0):
    global plot_items
    if time_end == 0.0:
        return
    for ind, plot_item in enumerate(plot_items):
        plot_item.setXRange(time_start, time_end)
        for item in plot_item.items:
            time_data, y_data = item.getOriginalDataset()
            if time_data is not None:
                selection = ( time_data < time_end ) & ( time_data > time_start )
                data = norm(y_data[selection]) if ind == 0 else y_data[selection]
                item.setData(x=time_data[selection], y=data)
crop_params = interact(crop, parent=initialization_params)
initialization_params.sigActivated.connect(crop_params.activate)

sub_mzi = plot_widget.addLayout()
sub_mzi.addLabel('MZI')
sub_mzi.nextRow()

plot_items = [pg.PlotItem(), pg.PlotItem()]
sub_mzi.addItem(plot_items[0])

plot_widget.nextRow()

sub_trans = plot_widget.addLayout()
sub_trans.addLabel('Transmission')
sub_trans.nextRow()
sub_trans.addItem(plot_items[1])

for ind, plot_item in enumerate(plot_items):
    view: pg.ViewBox = plot_item.getViewBox()
    view.setMouseEnabled(y=False)
#    view.setMouseMode(view.RectMode)
    plot_item.setDownsampling(auto=True, mode='peak')
    plot_item.setClipToView(True)
    view.disableAutoRange(view.YAxis)
    if ind == 1:
        view.setXLink(plot_items[0].getViewBox())

def get_plot_by_name(name, ind=None) -> pg.PlotDataItem:
    if ind is not None:
        return plot_items[ind].items[ [ item.name() for item in plot_items[ind].items ].index(name) ]
    for plot_item in plot_items:
        names = [ item.name() for item in plot_item.items ] 
        if name in names:
            return plot_item.items[ names.index(name) ]
    raise LookupError

def get_plot_item_by_plot(plot) -> pg.PlotItem:
    for plot_item in plot_items:
        if plot in plot_item.items:
            return plot_item
    raise LookupError

@interactor.decorate(ignores=['wlen_in_bw'])
def find_mzi_peaks(prominence=0.0, wlen_in_bw=400.0, distance_in_bw=3 / 4):
    global plot_items, mzi_peak_times
    assert mzi_bw != 0, 'Gotta initialize!'
    samples_per_mzi_bw = hz_to_samples(mzi_bw)
    item = get_plot_by_name('mzi')
    time, mzi_data = item.getOriginalDataset()

    if prominence == 0:
        ptp = []
        for chunk in np.array_split(mzi_data, 100):
            ptp.append(np.ptp(chunk))
        prominence = min(ptp) / 2

    peaks = find_peaks(
                mzi_data,
                prominence=prominence, 
                wlen=wlen_in_bw * samples_per_mzi_bw, 
                distance=distance_in_bw * samples_per_mzi_bw
            )[0]
    mzi_peak_times = time[peaks]

    try:
        peaks_item = get_plot_by_name('mzi_scatter')
        peaks_item.setData(x=mzi_peak_times, y=mzi_data[peaks])
    except LookupError:
        plot_items[0].addItem(
            pg.PlotDataItem(mzi_peak_times, mzi_data[peaks], name='mzi_scatter', pen=None, symbol='x', symbolPen=pg.mkPen("y"))
        )

    try:
        peaks_freq_item = get_plot_by_name('mzi_peak_freq')
        peaks_freq_item.setData(x=(mzi_peak_times[1:] + mzi_peak_times[:-1]) / 2, y=norm(1 / np.diff(mzi_peak_times)))
    except LookupError:
        plot_items[0].addItem(
            pg.PlotDataItem((mzi_peak_times[1:] + mzi_peak_times[:-1]) / 2, norm(1 / np.diff(mzi_peak_times)), name='mzi_peak_freq', pen=pg.mkPen('r'))
        )

class FrequencyAxis(pg.AxisItem):
    def __init__(self, time, freq, **kwargs):
        self.time_data = time
        self.samples_per_sec = len(time) / np.ptp(time)
        self.freq_data = freq
        c = 299792458 # m s^-1
        self.wl_scale = -np.ptp(c / freq) / np.ptp(time)
        self.func = self.make_fit(time[::100], 1e9 * c / freq[::100])
#        self.func = np.poly1d(np.polyfit(time[::100], 1e9 * c / freq[::100], 4))
        np.savez('temp.npz', time=time[::100], freq=freq[::100])
        self.initial_wavelength = c / freq[0]
        super().__init__(**kwargs)

    def tickStrings(self, values, scale, spacing):
        c = 299792458 # m s^-1
        #wavelengths = [ self.initial_wavelength + ( self.wl_scale * val ) for val in values ]
        wavelengths = self.func(np.array(values))
        return [f'{(c / (value * 1e-9)) / 1e12:.5f}\n{value:.3f}' for value in wavelengths]

    def make_fit(self, time, wavelength, threshold=0.1, direction=-1):
        middle_select = ( time > np.mean(time) - np.ptp(time) / 6 ) & ( time < np.mean(time) + np.ptp(time) / 6 )
        linear_fit = np.poly1d(np.polyfit(time[middle_select], wavelength[middle_select], 1))

        minus_linear = wavelength - linear_fit(time)

        if direction < 0:
            left_select = minus_linear < -threshold
            right_select = minus_linear > threshold
        else:
            left_select = minus_linear > threshold
            right_select = minus_linear < -threshold

        if np.sum(left_select):
            left_root = np.max(time[left_select])
            left_fit = np.poly1d(np.polyfit(time[left_select], minus_linear[left_select], 3))
        else:
            left_root = np.min(time)
            left_fit = np.poly1d([0])

        if np.sum(right_select):
            right_root = np.min(time[right_select])
            right_fit = np.poly1d(np.polyfit(time[right_select], minus_linear[right_select], 3))
        else:
            right_root = np.max(time)
            right_fit = np.poly1d([0])

        def fit(x):
            left = (left_fit + linear_fit)(x[x < left_root])
            middle = linear_fit(x[(x >= left_root) & (x <= right_root)])
            right = (right_fit + linear_fit)(x[x > right_root])
            return np.concat((left, middle, right))
        return fit

@interactor.decorate()
def find_frequency_from_mzi(fname='./mzi_calibration_all.npz'):
    global plot_items, mzi_peak_freq
    coefs = np.load(fname)['mzi_fit_coefficients']
    mode_to_freq = np.poly1d(coefs)
    roots = (mode_to_freq - c / start_wl).roots
    roots = roots[np.isreal(roots)].real
    start_mode = int(roots[np.argmin(np.abs(roots))])

    mode_time = mzi_peak_times
    mode_numbers = np.arange(len(mode_time))
    if velocity > 0:
        mode_numbers = mode_numbers[::-1]
    mode_numbers += start_mode
    freqs = mode_to_freq(mode_numbers)
    mzi_peak_freq = freqs

    for plot_item in plot_items:
        plot_item.clear()
        ''' This keeps the mzi peaks and stuff, but it slows everything down hella
        for plot in plot_item.items:
            _time, _data = plot.getOriginalDataset()
            mask = ( _time < np.max(mode_time) ) & ( _time > np.min(mode_time) )
            plot.setData(x=_time[mask], y=_data[mask])
        '''

    selection = ( time < np.max(mode_time) ) & ( time > np.min(mode_time) )
    full_freqs = np.interp(time[selection], mode_time, freqs, left=-1, right=-1)
    plot_items[0].addItem(
        pg.PlotDataItem(x=time[selection], y=mzi_data[selection], name='mzi')
    )
    plot_items[1].addItem(
        pg.PlotDataItem(x=time[selection], y=transmission_data[selection], name='transmission')
    )

    for plot_item in plot_items:
        plot_item.layout.removeItem(plot_item.getAxis('top'))
        axis = FrequencyAxis(time=time[selection], freq=full_freqs, orientation='top', parent=plot_item)
        axis.setLabel('Freq and Wavelength')
        axis.linkToView(plot_item.vb)
        plot_item.axes['top']['item'] = axis
        plot_item.layout.addItem(axis,1,1)
        plot_item.setDownsampling(auto=True, mode='peak')
        plot_item.setClipToView(True)

@interactor.decorate()
def smooth_transmission(sigma=5):
    item = get_plot_by_name('transmission')
    time_data, data = item.getOriginalDataset()
    smoothed_data = gaussian_filter1d(data, sigma)
    item.setData(x=time_data, y=smoothed_data)

@interactor.decorate()
def find_transmission_peaks(prominence=0.001, wlen_in_bw=1, distance_in_bw=0.1, Q_min=1e5, fsr=res_fsr, find_fits=False):
    global plot_items, transmission_peaks_freq, transmission_peaks_Q, resonance_info, res_fsr
    res_fsr = fsr
    samples_per_resonance = hz_to_samples(fsr)
    fsr_estimate = hz_to_samples((c / start_wl) / Q_min)

    item = get_plot_by_name('transmission')
    x_data, transmission_data = item.getOriginalDataset()
    freq = get_plot_item_by_plot(item).axes['top']['item'].freq_data
    peak_data = find_peaks((1 - transmission_data), 
                       prominence=prominence, 
                       distance=samples_per_resonance * distance_in_bw, 
                       wlen=samples_per_resonance * wlen_in_bw, 
                       width=[0, fsr_estimate]
    )

    if find_fits:
        #peaks, peak_omega, peak_Q = refine_transmission_peaks(item, freq, peak_data, samples_per_resonance)
        resonance_info = refine_transmission_peaks(item, freq, peak_data, samples_per_resonance)
    else:
        peaks = peak_data[0]
        peak_omega = freq[peaks]
        peak_Q = []
        for center_ind, width in zip(peak_data[0], peak_data[1]['widths']):
            fwhm = np.abs(freq[center_ind - int(width) // 2] + freq[center_ind - int(width) // 2])
            peak_Q.append(freq[center_ind] / fwhm)
        peak_Q = np.array(peak_Q)

        resonance_info = pd.DataFrame({
            'freq': peak_omega,
            'Q_loaded': peak_Q,
            'peak_ind': peaks.astype(int)
        })

    try:
        item = get_plot_by_name('transmission_scatter')
        item.clear()
    except LookupError:
        item = pg.PlotDataItem(name='transmission_scatter', pen=None, symbol='x', symbolPen=pg.mkPen("y"))
        plot_items[1].addItem(item)
#    (item.scatter)
    transmission_peaks_freq = resonance_info['freq']
    transmission_peaks_Q = resonance_info['Q_loaded']
    #item.setData(x_data[peaks], transmission_data[peaks])
    print(resonance_info)
    item.setData(x_data[resonance_info['peak_ind']], transmission_data[resonance_info['peak_ind'].to_numpy()])

def lorentzian(omega, omega0, gamma, m, b):
    return m * (1 - (gamma / 2) ** 2 / ( (omega - omega0) ** 2 + (gamma / 2) ** 2 )) + b

# def lorentzian(omega, omega_0, alpha_i, theta, Ein):
#     alpha = ( alpha_i + theta ) / 2
#     #return np.abs( ( 1 - theta / ( alpha + 1j * (omega - omega_0) ) ) * Ein ) ** 2 + a * ( omega - omega_0 )
#     return np.abs( ( 1 - theta / ( alpha + 1j * (omega - omega_0) ) ) * Ein ) ** 2

def find_resonance_information(frequency, transmission, fwhm_guess) -> dict:
    freq_0_guess = frequency[np.argmin(transmission)]
    Ein_guess = np.sqrt(np.max(transmission))
#    extinction_guess = np.ptp(transmission) / np.max(transmission)
#    alpha_guess = fwhm_guess / 2
#    theta_guess = alpha_guess * ( 1 + np.sqrt(1 - extinction_guess) )
#    alpha_i_guess = alpha_guess * ( 1 - np.sqrt(1 - extinction_guess) )
    theta_guess = fwhm_guess / 2
    alpha_i_guess = fwhm_guess / 2

    fit, cov = curve_fit(lorentzian, frequency, transmission, [freq_0_guess, alpha_i_guess, theta_guess, Ein_guess])
    freq_0, alpha_i, theta, Ein = fit

#    print(f'{alpha_i=:.2e}, {alpha_i_guess=:.2e}, {theta=:.2e}, {theta_guess=:.2e}')

    delta_alpha = ( alpha_i - theta ) / 2
    alpha = ( alpha_i + theta ) / 2
    extinction = 1 - delta_alpha ** 2 / alpha ** 2
    Q_loaded = freq_0 / (theta + alpha_i)

    resonance = {
        'freq': freq_0,
        'alpha': alpha,
        'delta_alpha': delta_alpha,
        'extinction': extinction,
        'Q_loaded': Q_loaded,
        'fit': fit
    }
    return resonance

def refine_transmission_peaks(item, freq, peak_data, samples_per_resonance):
    time, transmission = item.getOriginalDataset()
    peak_center = peak_data[0]
    peak_widths = peak_data[1]['widths']
    skirt = samples_per_resonance // 20

    fits = np.empty((0,))
    fits_time = np.empty((0,))
    resonances = []
#    new_peaks = []
#    peak_omega = []
#    peak_Q = []

    for ind, peak in enumerate(peak_center):
        peak_width = int(peak_widths[ind])
        skirt = 4 * peak_width
        if peak < skirt:
            continue
        _freq = freq[peak - skirt:peak + skirt + 1]
        _trans = transmission[peak - skirt:peak + skirt + 1]
#        res_info = find_resonance_information(_freq, _trans, peak_width)
        gamma_est = np.sqrt(8) * np.abs(freq[peak + peak_width // 2] - freq[peak - peak_width // 2])

        try:
            fit_parameters = curve_fit(lorentzian,
                                       _freq,
                                       _trans,
                                       [freq[peak], gamma_est, 1, 0],
                                       full_output=False,
            )
        except ValueError:
            print(_freq)
            print(_trans)
            import matplotlib.pyplot as plt
            plt.plot(_freq, _trans)
            plt.show()
            continue
        except RuntimeError:
            import matplotlib.pyplot as plt
            plt.plot(_freq, _trans)
            plt.show()
            continue
        omega0, gamma, _, _ = fit_parameters[0]
        fwhm = gamma / np.sqrt(8)
        Q = omega0 / fwhm
        omega0_ind = np.argmin(np.abs(_freq - omega0)) + peak - skirt
        res_info = {
            'freq': omega0,
            'Q_loaded': Q,
            'peak_ind': omega0_ind,
            'fit': fit_parameters[0]
        }
#        peak_omega.append(omega0)
#        peak_Q.append(Q)
#        new_peaks.append(omega0_ind)


        _time = time[peak - skirt:peak + skirt + 1]
        _fit = lorentzian(_freq, *fit_parameters[0])

        peak_ind = np.argmin(np.abs(_freq - res_info['freq'])) + peak - skirt
        res_info['peak_ind'] = peak_ind

        resonances.append(res_info)
        _time = time[peak - skirt:peak + skirt + 1]
        _fit = lorentzian(_freq, *res_info['fit'])
        fits = np.append(fits, _fit)
        fits_time = np.append(fits_time, _time)
    try:
        item = get_plot_by_name('transmission_fits')
        item.clear()
    except LookupError:
        item = pg.PlotDataItem(name='transmission_fits', pen=pg.mkPen("r"))
        plot_items[1].addItem(item)
    item.setData(x=fits_time, y=fits)

    return pd.DataFrame.from_dict(resonances, orient='columns')
    #return np.array(new_peaks), np.array(peak_omega), np.array(peak_q)

@interactor.decorate()
def dump_results(fname=''):
    np.savez(
        fname,
        transmission_peak_freq=transmission_peaks_freq,
        transmission_peak_Q=transmission_peaks_Q,
        mzi_peak_times=mzi_peak_times,
        mzi_peak_freq=mzi_peak_freq,
        velocity=velocity,
        start_wl=start_wl,
        fsr=res_fsr
    )

if __name__ == '__main__':
    pg.exec()
