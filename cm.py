import os
import sys
import numpy as np
import pandas as pd
import nidaqmx
import bokeh
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import nidaqmx.system as system


def data_path():
    return os.getcwd() + "\\Data\\"


class CurrentMonitor:
    """ Configures a set of channels to monitor single channels for DC Current
    """

    def __init__(self, idx):
        # Manual definition of test set hardware
        #self.signals = ['Iwe1a', 'Iwe2a', 'Iiea', 'Iwe1b', 'Iwe2b', 'Iieb']
        #self.physical_channels = ['ai0', 'ai1', 'ai5', 'ai2', 'ai3', 'ai7']
        self.signals = ['Iwe1a', 'Iwe2a', 'Iiea']
        self.physical_channels = ['ai0', 'ai1', 'ai2']
        self.sense_resistors = [100, 100, 100]  # position dependent upon sense resistors
        self.amp_gains = [1, 1, 1]
        self.num_channels = 3
        # Test Set Configure end
        try:
            self.name = system.System.local().devices[idx].name
            self.product_type = system.System.local().devices[idx].product_type
            self.serial_number = system.System.local().devices[idx].dev_serial_num
            self.max_multi_chan_rate = system.System.local().devices[idx].ai_max_multi_chan_rate
            self.max_single_chan_rate = system.System.local().devices[idx].ai_max_single_chan_rate
            self.ai_voltage_rngs = system.System.local().devices[idx].ai_voltage_rngs
        except nidaqmx.DaqError as e:
            print(e)
            sys.exit(1)
        print(f'{self.product_type} found named:{idx}')
        self.max_sample_rate_per_channel = int(self.max_single_chan_rate / self.num_channels)
        self.sample_rate_per_channel = self.max_sample_rate_per_channel  # set to max sample rate unless overridden
        self.physical_channel_string = ', '.join([self.name + '/' + i for i in self.physical_channels])
        self.physical_channel_by_signal = dict(zip(self.signals, self.physical_channel_string.split(',')))
        self.v_offset = [0]
        self.v_to_ua = [1e6 / (gain * res) for gain, res in zip(self.amp_gains, self.sense_resistors)]
        self.ua_to_v = [(gain * res) for gain, res in zip(self.amp_gains, self.sense_resistors)]
        self.v_offset_by_signal = dict(zip(self.signals, self.v_offset))
        self.v_to_ua_by_signal = dict(zip(self.signals, self.v_to_ua))
        self.ua_to_v_by_signal = dict(zip(self.signals, self.ua_to_v))
        self.amp_gain_by_signal = dict(zip(self.signals, self.amp_gains))
        self.sense_resistor_by_signal = dict(zip(self.signals, self.sense_resistors))
        self.mean_na_raw = None
        self.std_na_raw = None
        self.meas_ua_raw = None
        self.meas_ua = None
        self.mean_na = None
        self.std_na = None

        print(f'Signals: {self.signals} on {self.physical_channels} are configured')

    def get_device_accuracy(self, voltage_range, measure="voltage", resistor_value=47):
        """Gets accuracy of first NI device connected in list
        'voltage_range' is the range that is used in the device. 20 not implemented with divider yet!
        'measure' is either 'VOLTAGE' or 'CURRENT' where current needs a sense resistor
        'resistor_value' depends on test set sense resistor
        """
        df = None
        if "6361" in self.product_type:
            df = pd.DataFrame(np.array([[10, 315, 1660],
                                        [5, 157, 870],
                                        [2, 64, 350],
                                        [1, 38, 190],
                                        [0.5, 27, 100],
                                        [0.2, 21, 53],
                                        [0.1, 17, 33]]),
                              columns=['Range', 'Noise', 'Accuracy'])
        elif "6251" in self.product_type:
            df = pd.DataFrame(np.array([[10, 280, 1920],
                                        [5, 140, 1010],
                                        [2, 57, 410],
                                        [1, 32, 220],
                                        [0.5, 21, 130],
                                        [0.2, 16, 74],
                                        [0.1, 15, 52]]),
                              columns=['Range', 'Noise', 'Accuracy'])
        elif "6281" in self.product_type:
            df = pd.DataFrame(np.array([[10, 280, 1920],  # TODO values are wrong for this device
                                        [5, 140, 1010],
                                        [2, 57, 410],
                                        [1, 32, 220],
                                        [0.5, 21, 130],
                                        [0.2, 16, 74],
                                        [0.1, 15, 52]]),
                              columns=['Range', 'Noise', 'Accuracy'])
        elif "6218" in self.product_type:
            df = pd.DataFrame(np.array([[10, 280, 1920],  # TODO values are wrong for this device
                                        [5, 140, 1010],
                                        [2, 57, 410],
                                        [1, 32, 220],
                                        [0.5, 21, 130],
                                        [0.2, 16, 74],
                                        [0.1, 15, 52]]),
                              columns=['Range', 'Noise', 'Accuracy'])
        df.set_index("Range", inplace=True)
        accuracy = df.loc[voltage_range, 'Accuracy']
        if measure == 'current':
            accuracy /= resistor_value
        return round(accuracy, 2)

    def measure_channel_current(self, signal, maximum_current_ua=10.0, duration_ms=1000, low_pass_enable=True):
        """Measure the current on one channel
        :param signal:
        :param maximum_current_ua:
        :param duration_ms:
        :param low_pass_enable:
        :return:
        """
        physical_channel = self.physical_channel_by_signal[signal]
        maximum_voltage = 1e-6 * maximum_current_ua * self.sense_resistor_by_signal[signal]
        voltage_range = self.get_minimum_voltage_range(maximum_voltage)
        samples = int(self.max_single_chan_rate * duration_ms / 1000)
        # print(f'Signal {signal} is being acquired for {duration_ms}mS at a SR{self.max_single_chan_rate}Hz '
        #       f'for currents less than {maximum_current_ua}uA'
        #       f' and a voltage drop not exceeding {voltage_range} volts')
        with nidaqmx.Task() as ai_task:
            try:
                ai_task.ai_channels.add_ai_voltage_chan(physical_channel,
                                                        max_val=voltage_range, min_val=-voltage_range,
                                                        terminal_config=TerminalConfiguration.DIFFERENTIAL)
                if '6281' in self.product_type: # has filter
                    ai_task.ai_channels.all.ai_lowpass_enable = low_pass_enable
                ai_task.timing.cfg_samp_clk_timing(self.max_single_chan_rate, samps_per_chan=samples,
                                                   sample_mode=AcquisitionType.FINITE)
                ai_task.start()
                da_v = np.array(ai_task.read(number_of_samples_per_channel=samples, timeout=duration_ms / 1000 + 1))
            except nidaqmx.DaqWarning as e:
                print('Daq Warning caught as exception: {0}\n'.format(e))
                assert e.error_code == 200015
                sys.exit(1)
            except nidaqmx.DaqError as e:
                print('Daq Error caught as exception: {0}\n'.format(e))
                assert e.error_code == 200015
                sys.exit(1)
        self.meas_ua_raw = da_v * self.v_to_ua_by_signal[signal]
        self.mean_na_raw = 1000 * np.mean(self.meas_ua_raw)   # nano amperes
        self.std_na_raw = 1000 * np.std(self.meas_ua_raw)

    def trim_invalid_biphasic_pulses(self, threshold_ua=100, sample_margin=10, polarity_negative=False):
        """ Trims first and last valid biphasic sequences as determined by edges at a threshold
        used when there is a pulse train signal
        :param polarity_negative: sign must be switched
        :param sample_margin: samples before and after trigger
        :param threshold_ua:
        :return:
        """
        a = self.meas_ua_raw
        if polarity_negative:
            a = -a

        pos_rising_edges = np.flatnonzero((a[:-1] < threshold_ua) & (a[1:] > threshold_ua)) + 1
        pos_falling_edges = np.flatnonzero((a[:-1] > threshold_ua) & (a[1:] < threshold_ua)) + 1
        neg_falling_edges = np.flatnonzero((-a[:-1] < threshold_ua) & (-a[1:] > threshold_ua)) + 1
        neg_rising_edges = np.flatnonzero((-a[:-1] > threshold_ua) & (-a[1:] < threshold_ua)) + 1
        print(f'min+^:{min(pos_rising_edges)} max+^:{max(pos_rising_edges)}')
        print(f'min+v:{min(pos_falling_edges)} max+v:{max(pos_falling_edges)}')
        print(f'min-v:{min(neg_rising_edges)} max+v:{max(neg_rising_edges)}')
        print(f'min-^:{min(neg_rising_edges)} max-^:{max(neg_rising_edges)}')

        # find the first valid biphasic sequence
        idx = 0
        while not (pos_rising_edges[idx] < pos_falling_edges[idx] < neg_falling_edges[idx] < neg_rising_edges[idx]):
            idx += 1
        # lowest_index = min(min(pos_rising_edges), min(pos_falling_edges), min(neg_falling_edges), min(neg_rising_edges))
        lowest_index = min(neg_falling_edges)
        print(f'first valid index{lowest_index}')

        # if the edges lists are not the same size then a full biphasic sequence was not finished so trim
        min_len = min(len(pos_rising_edges), len(pos_falling_edges), len(neg_falling_edges), len(neg_rising_edges))
        pos_rising_edges = pos_rising_edges[:min_len]
        pos_falling_edges = pos_falling_edges[:min_len]
        neg_falling_edges = neg_falling_edges[:min_len]
        neg_rising_edges = neg_rising_edges[:min_len]

        # then find the highest index captured
        # highest_index = max(max(pos_rising_edges), max(pos_falling_edges), max(neg_falling_edges), max(neg_rising_edges))
        highest_index = max(neg_rising_edges)
        print(f'last valid index {highest_index}')
        pre_trig_loc = lowest_index - sample_margin if (lowest_index - sample_margin) >= 0 else 0
        post_trig_loc = highest_index + sample_margin if (highest_index + sample_margin) <= (len(a) - 1) else (
                    len(a) - 1)
        print(f'pre_post_trig {pre_trig_loc} - {post_trig_loc}')
        self.meas_ua = a[pre_trig_loc:post_trig_loc]
        self.mean_na = 1000 * np.mean(self.meas_ua)
        self.std_na = 1000 * np.std(self.meas_ua)

    def get_minimum_voltage_range(self, max_voltage):
        """ Given maximum expected voltage determine lowest feasible voltage range
        :return is the lowest possible range from nidaq device
        """
        a = np.abs(np.array(self.ai_voltage_rngs))
        return a[np.argmax(a > max_voltage)]


def filter_60hz_notch(na, fs=666666, f0=60, q=30):
    from scipy import signal
    b, a = signal.iirnotch(f0, q, fs)
    y = signal.filtfilt(a, b, na)
    return y


def list_nidaq_devices():
    print(list(system.System.local().devices))


def measure_currents_on_channel_for_conditions(measurement, signal='I1', file_name="Current_Check_Data.csv"):
    """ For sets of amplitudes, pw, rpr, frequencies measure currents and build table
    :param measurement: DC Measure class instance must be passed in
    :param signal:
    :param file_name:
    :return:
    """
    try:
        myfile = open(file_name, mode='w')
        myfile.close()
    except IOError:
        print(f"Could not open file! Please close Excel!")
        return []
    columns = ['AC_Current', 'AC_Frequency', 'AC_PW', 'AC_RPR', 'Load', 'Mean_nA', 'StdDev_nA']
    df = pd.DataFrame(columns=columns)
    # Lists of combinations to check - all lists need to be same length
    ac_currents_ma = [0, 5000, 5000, 5000, 5000]
    ac_frequencies = [50, 50, 100, 400, 1000]
    ac_pws = [240, 240, 240, 240, 240]
    ac_rprs = [4, 4, 4, 4, 4]
    for ac_current_ma, ac_pw, ac_rpr, ac_frequency in zip(ac_currents_ma, ac_pws, ac_rprs, ac_frequencies):
        max_frequency = 1e6 / (ac_pw * (1 + ac_rpr) + 30)
        print(f'Max frequency: {max_frequency}Hz')
        if max_frequency < ac_frequency:
            ac_frequency = max_frequency
        print(f'Set amp to:{ac_current_ma}  PW to:{ac_pw} RPR to:{ac_rpr} frequency to:{ac_frequency} ')
        myi = input(f's to skip, q to quit, return to continue')
        if myi == 's':
            print('Condition skipped')
        elif myi == 'q':
            break
        else:
            measurement.measure_channel_current(signal, maximum_current_ua=ac_current_ma)
            print(f'AC Current:{ac_current_ma:.0f}mA Mean:{measurement.mean_na_raw:.0f}nA '
                  f'StdDev:{measurement.std_na_raw:.0f}nA')
            d = {
                'AC_Current': ac_current_ma,
                'AC_Frequency': ac_frequency,
                'AC_PW': ac_pw,
                'AC_RPR': ac_rpr,
                'Mean_nA': measurement.mean_na_raw,
                'StdDev_nA': measurement.std_na_raw
            }
            df = df.append(d, ignore_index=True)
    df.to_csv(file_name, index=False)
    return df


def plot_numpy_array(a, plot_title='Current', y_axis_label='Current(uA)'):
    """
    quick plot of numpy array
    """
    from bokeh.plotting import figure, output_file, show
    output_file(data_path() + 'SimpleNumpyPlot.html')
    p = figure(title=plot_title, y_axis_label=y_axis_label)
    x = np.linspace(0, len(a) - 1, len(a))
    p.line(x, a, color='blue')
    show(p)

def plot_numpy_array_charge(a, sr, plot_title='Current', y_axis_label='Current(uA)',):
    """
    quick plot of numpy array
    """
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import LinearAxis, Range1d
    import numpy as np

    output_file(data_path() + 'SimpleNumpyPlot.html')
    p = figure(title=plot_title, y_axis_label=y_axis_label)
    x = np.linspace(0, len(a) - 1, len(a))*(1/sr)
    p.line(x, a, color='blue')

    b = np.cumsum(a)
    b = b / sr

    p.extra_y_ranges = {"Charge": Range1d(start=min(b), end=max(b))}
    p.add_layout(LinearAxis(y_range_name="Charge", axis_label='Charge (uC)'), 'right')

    p.line(x, b, y_range_name="Charge", color='red')
    show(p)