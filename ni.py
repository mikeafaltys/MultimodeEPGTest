"""
Instructions:
    use requirements.txt to install all required python dependencies
    Manual Set Path below
    Use simple_capture_plot() for basic operations
"""
# from PyDAQmx import DAQError
from collections import namedtuple
import os
import sys
import numpy as np
import pandas as pd
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.constants import Slope, TaskMode
import nidaqmx.system as system


def data_path():
    return os.getcwd() + "\\Data\\"


test_document = 'QP10330 rev.0'


class TestSetConfig:
    def __init__(self):
        self.test_software = 'CS10XXX rev. X'
        self.test_software_build = '0.0.1'
        self.cpa_software = 'CS10XXX rev. X'
        self.cpa_software_build = '0.49.74433.11531'
        self.epg_sn = 'E00011'
        self.epg_model = 'EPGv2'
        self.daq_tool = 'T00xx-0X'
        self.daq_adapter = 'T00XX-0X'


test_set_config = TestSetConfig()


class NidaqHardware:
    def __init__(self, idx, rwe1=49.9, rwe2=49.9, rie=49.9, rbat=0.28):
        try:
            self.name = system.System.local().devices[idx].name
            self.max_multi_chan_rate = system.System.local().devices[idx].ai_max_multi_chan_rate
            self.max_single_chan_rate = system.System.local().devices[idx].ai_max_single_chan_rate
            self.product_type = system.System.local().devices[0].product_type
            self.ai_voltage_rngs = system.System.local().devices[0].ai_voltage_rngs
            self.ai_channel_names = system.System.local().devices[0].ai_physical_chans.channel_names
            self.di_ports = system.System.local().devices[0].di_ports.channel_names
            self.do_ports = system.System.local().devices[0].do_ports.channel_names
            self.terminals = system.System.local().devices[0].terminals
            self.serial_number = system.System.local().devices[0].dev_serial_num
            self.rwe1 = rwe1
            self.rwe2 = rwe2
            self.rie = -rie
            self.rbat = rbat
        except:
            print("Error Connecting to Nidaq Device")
            sys.exit(1)


daq = NidaqHardware(0)  # MAKE INSTANCE OF NidaqHardware Class

# Signals are configured as a global - which probably means a data class is a better choice than namedtuple
Signal = namedtuple('Signal', 'Name Chan Axis Type Mult Color Dash')
iwe1 = Signal(Name='Iwe1', Chan='ai0', Axis='left', Type='current', Mult=1e6 / daq.rwe1, Color='red', Dash='solid')
iwe2 = Signal(Name='Iwe2', Chan='ai2', Axis='left', Type='current', Mult=1e6 / daq.rwe2, Color='black',
              Dash='solid')
iie = Signal(Name='Iie', Chan='ai4', Axis='left', Type='current', Mult=1e6 / daq.rie, Color='green', Dash='solid')
ibat = Signal(Name='Bat_mA', Chan='ai7', Axis='left', Type='current', Mult=-1e3 / daq.rbat, Color='magenta',
              Dash='solid')
vwe1 = Signal(Name='Vwe1_ie', Chan='ai1', Axis='right', Type='voltage', Mult=1000, Color='orange', Dash='solid')
vwe2 = Signal(Name='Vwe2_ie', Chan='ai3', Axis='right', Type='voltage', Mult=1000, Color='cyan', Dash='solid')
v14 = Signal(Name='V14_0', Chan='ai14', Axis='right', Type='voltage', Mult=1000, Color='lime', Dash='solid')

#  DEFINE SIGNALS TO BE UTILIZED AS GLOBAL
signals = (iwe1, iwe2, iie, vwe1, vwe2)  # tuple of signals to print (CURRENTS MUST BE FIRST)


def simple_capture_plot(plot_title='Device_Mode_Params', sample_rate=100000, samples=20000, window=10,
                        max_amplitude_ua=15000):
    """Captures and plots signals in the signal_spec for demonstration purposes
    :param: sample_rate where the maximum is defined as 1e6/(number of channels used)
    :param: samples is the number of samples that will be captured
    :param: applies a running window that averages over a number of samples
        window=1: 5uA peak-to-peak noise (Irving over 6" RJ-45 to electrochemical cell 3" lead)
        window=10 2uA peak-to-peak noise
        window=20 1.25uA peak-to-peak noise
    :return: df - output result with pandas data frame
    """
    from bokeh.io import output_file, show
    from bokeh.plotting import figure
    from bokeh.models import Label

    # Define physical acquisition signals i.e. 'Dev1/ai0, Dev1/ai2'
    current_channels = ', '.join([daq.name + '/' + i.Chan for i in signals if i.Type == 'current'])
    voltage_channels = ', '.join([daq.name + '/' + i.Chan for i in signals if i.Type == 'voltage'])
    current_range = get_minimum_current_range(max_amplitude_ua)
    voltage_range = 10.0  # voltage resolution is maxed because voltages are not critical

    with nidaqmx.Task() as ai_task:
        ai_task.ai_channels.add_ai_voltage_chan(physical_channel=current_channels,
                                                max_val=current_range, min_val=-current_range,
                                                terminal_config=TerminalConfiguration.DIFFERENTIAL)
        try:
            ai_task.ai_channels.add_ai_voltage_chan(physical_channel=voltage_channels,
                                                    max_val=voltage_range, min_val=-voltage_range,
                                                    terminal_config=TerminalConfiguration.DIFFERENTIAL)
        except:
            print(f'DAQmx Error')
        ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples, sample_mode=AcquisitionType.FINITE)
        ai_task.start()
        da = np.array(ai_task.read(number_of_samples_per_channel=samples,
                                   timeout=samples / sample_rate + 1 + 100)).transpose()
    # transform voltages to currents
    multiplier = [i.Mult for i in signals]
    dt = np.linspace(0, samples / sample_rate, samples)
    da = da * np.array(multiplier)

    output_file(data_path() + plot_title + '.html')
    p = figure(title=plot_title, x_axis_label='Time(s)', y_axis_label='Voltage/Current (mV/uA)')
    citation_label = 'Average Currents: '
    # sin
    for i, signal in enumerate(signals):
        da[:, i] = np.convolve(da[:, i], np.ones((window,)) / window, mode='same')  # running average filter
        if signal.Type == 'current':
            citation_label += f'{signal.Name}: {da[:, i].mean():.2f}uA '
        p.line(dt, da[:, i], legend_label=signal.Name, line_color=signal.Color, line_dash=signal.Dash)

    citation = Label(x=25, y=70, x_units='screen', y_units='screen', text=citation_label, render_mode='css',
                     border_line_color='black', border_line_alpha=0.7,
                     background_fill_color='pink', background_fill_alpha=0.8, text_font_size='10pt')
    p.add_layout(citation)
    p.legend.click_policy = "hide"
    show(p)  # displays plot in default web browser

    columns = [i.Name for i in signals]
    df = pd.DataFrame(data=da[0:, 0:], index=dt[0:], columns=columns)
    return df


def ac_frequency(pw_ua=60, rpw=8, ipi_us=30, dt1_us=500):
    return 1e6 / (pw_ua * (1 + rpw) + ipi_us + dt1_us)


def crossings(a, slice_level):
    b = np.array(a)
    neg_edges = np.flatnonzero((b[:-1] > slice_level) & (b[1:] < slice_level)) + 1
    pos_edges = np.flatnonzero((b[:-1] < slice_level) & (b[1:] > slice_level)) + 1
    print(f'trigger level:{slice_level}  Pos: {pos_edges}  Neg: {neg_edges}')
    return pos_edges, neg_edges


def extract_ac_pulse_parameters(b, polarity='anodic', amplitude_ua=5000, sample_rate=2e6):
    """ Extract the pulse params from spec - tedious
    :param b:  1-dimensional list or array
    :param polarity: 'Anodic' indicates PW is anodic and RPW is cathodic, whereas 'Cathodic' is opposite
    :param amplitude_ua:
    :param sample_rate: S
    :return: data in data dictionary format
    """
    high = 0.8  # 80%
    low = 0.2  # 20%
    rpw = 8
    amp = amplitude_ua

    amp_sign = 1  # Pulse timings are setup for anodic pw, just flip the signs on cathodic interface
    if polarity == 'anodic':
        a = np.array(b)
    else:
        a = -np.array(b)
        amp_sign = -1

    # Use data dictionary to build capture points at 80 and 20% on PW and RPW
    crosses = {'PW-H-POS': 1, 'PW-H-NEG': 1, 'PW-L-POS': 1, 'PW-L-NEG': 1,
               'RPW-H-POS': 1, 'RPW-H-NEG': 1, 'RPW-L-POS': 1, 'RPW-L-NEG': 1}
    slices = {'PW-H-POS': high * amp, 'PW-H-NEG': high * amp, 'PW-L-POS': low * amp, 'PW-L-NEG': low * amp,
              'RPW-H-POS': -amp * high / rpw, 'RPW-H-NEG': -amp * high / rpw, 'RPW-L-POS': -amp * low / rpw,
              'RPW-L-NEG': -amp * low / rpw}

    for key in slices.keys():  # collect zero crossings
        if 'NEG' in key:
            crosses[key] = np.flatnonzero((a[:-1] > slices[key]) & (a[1:] < slices[key])) + 1
        else:
            crosses[key] = np.flatnonzero((a[:-1] < slices[key]) & (a[1:] > slices[key])) + 1

    us = 1.0e6 / sample_rate
    # extract all parameters
    data = {'Period': sample_rate / (crosses['PW-H-POS'][1] - crosses['PW-H-POS'][0]),
            'PW-H': (crosses['PW-H-NEG'][0] - crosses['PW-H-POS'][0]) * us,
            'PW-L': (crosses['PW-L-NEG'][0] - crosses['PW-L-POS'][0]) * us,
            'PW-AMP': amp_sign * np.mean(a[crosses['PW-H-POS'][0]: crosses['PW-H-NEG'][0]]),
            'IPI': (crosses['RPW-L-NEG'][0] - crosses['PW-L-NEG'][0]) * us,
            'PW-RT': (crosses['PW-H-POS'][0] - crosses['PW-L-POS'][0]) * us,
            'PW-FT': (crosses['PW-L-NEG'][0] - crosses['PW-H-NEG'][0]) * us,
            'RPW-H': (crosses['RPW-H-POS'][0] - crosses['RPW-H-NEG'][0]) * us,
            'RPW-AMP': amp_sign * np.mean(a[crosses['RPW-H-NEG'][0]: crosses['RPW-H-POS'][0]]),
            'RPW-L': (crosses['RPW-L-POS'][0] - crosses['RPW-L-NEG'][0]) * us,
            'RPW-RT': (crosses['RPW-L-POS'][0] - crosses['RPW-H-POS'][0]) * us,
            'RPW-FT': (crosses['RPW-H-NEG'][0] - crosses['RPW-L-NEG'][0]) * us,
            'DT1': (crosses['PW-L-POS'][1] - crosses['RPW-L-POS'][0]) * us,
            'DT1-AMP': amp_sign * np.mean(a[crosses['RPW-L-POS'][0]:crosses['PW-L-POS'][1]])
            }

    return data


def test_ac(amplitude_ua, pw_ua=60, rpw=8, ipi_us=30, dt1_us=510, load_resistance=510, cycles=1, filter_window=1):
    """ Tests AC Capability
    :param amplitude_ua:
    :param pw_ua:
    :param rpw:
    :param ipi_us:
    :param dt1_us:
    :param load_resistance:
    :param cycles:
    :param filter_window:
    :return:
    """
    test_name = f'AC {amplitude_ua:.0f}uA_{pw_ua}_{ipi_us}_{rpw}x_{dt1_us}us'
    columns = ['Iwe1', 'Iwe2', 'Iie', 'Vwe1', 'Vwe2']
    channels = ['ai0', 'ai2', 'ai4', 'ai1', 'ai3']
    df = pd.DataFrame(columns=columns)
    sample_rate = daq.max_single_chan_rate
    period_us = int(pw_ua * (1 + rpw) + ipi_us + dt1_us)
    samples_cycle = int(sample_rate * 1e-6 * period_us)
    pre_trigger_samples = int(samples_cycle / 10)
    samples_to_capture = samples_cycle * cycles + 2 * pre_trigger_samples
    print(f'Period: {period_us}  Samples:{samples_to_capture} SR: {sample_rate} '
          f'Duration:{samples_to_capture / sample_rate}')
    # Figure out what is on channel
    for channel, column in zip(channels, columns):
        dv = get_single_channel(channel=channel, voltage_range=10.0, sample_rate=sample_rate,
                                samples=samples_to_capture,
                                triggered=False)
        max_measured = dv.max()
        min_measured = dv.min()
        abs_max_measured = max(abs(max_measured), abs(min_measured))
        voltage_range = get_minimum_voltage_range(abs_max_measured)

        if abs(max_measured) > abs(min_measured):
            trigger_level = 0.9 * max_measured
        else:
            trigger_level = 0.9 * min_measured

        print(f'Waiting for trigger at {trigger_level}')
        dv = get_single_channel(channel=channel, voltage_range=voltage_range, sample_rate=sample_rate,
                                samples=samples_to_capture,
                                triggered=True, trigger_level=trigger_level, pre_trigger_samples=pre_trigger_samples)

        d = (dv * 1e3) if 'Vwe' in column else (dv / 49.9) * 1e6
        print(f'col:{column} chan:{channel} range:{voltage_range} max:{max(d)} min:{min(d)}')
        df[column] = d

    dt = np.linspace(0, samples_to_capture / sample_rate, samples_to_capture)
    df['Time'] = dt

    plot_data_frame(df, test_name, "Amplitude (uA/mV")

    data = extract_ac_pulse_parameters(df['Iwe1'], amplitude_ua, rpw=rpw, sample_rate=sample_rate)
    df_results = pd.DataFrame.from_dict(data, orient='index', columns=['Iwe1-Meas'])
    data = extract_ac_pulse_parameters(-df['Iwe2'], amplitude_ua, rpw=rpw, sample_rate=sample_rate)
    df_temp = pd.DataFrame.from_dict(data, orient='index', columns=['Iwe2-Meas'])

    # test_results = TestResultZeroHz(test_name, amplitude_ua, test_data, 1000 / period1)
    # test_results.print_report()
    df_results['Iwe2-Meas'] = df_temp['Iwe2-Meas']

    return df, df_results


"""
    dt, da = get_single_channel(channel='ai4', sample_rate=sample_rate, samples=samples_cycle_est,
                                maximum_current_ua=amplitude_ua)

    # 1 Get Waveform and filter
    df_raw_meas = get_iv_data(sample_rate, samples_cycle_est, amplitude_ua)

    capture_boundary = int(samples_cycle_est / 10.0)
    dft, frequency, pulsewidth = \
        _software_trigger(df_raw_meas, 'Iwe1', sample_rate,
                          slice_level=amplitude_ua / 4,
                          post_trigger_samples=samples_cycle_est + capture_boundary,
                          pre_trigger_samples=capture_boundary, negative_edge_trigger=True)

    # df_raw_meas[:] = df_raw_meas[:].rolling(window=filter_window, min_periods=1, win_type='blackmanharris').mean()
    # Measure frequency measurement so reference can be created at same frequency
    # period1 = measure_period(df_raw_meas, 'Iwe1', 0, cycles)
    # period2 = measure_period(df_raw_meas, 'Iwe2', 0, cycles)
    # samples_in_cycle = int((period1 + period2) / 2)
    # 2 Calculate Reference Waveform
    # df_ref = build_zerohz_waveform_from_template(amplitude_ua=amplitude_ua, samples=samples_in_cycle,
    #                                              load_resistance=load_resistance, offset_ua=0)
    # 3 Extract period of measure based upon best correlation to reference (more reliable than edge or level triggering
    # df_meas = extract_measurement_that_matches_ref(df_ref, df_raw_meas, 'Iwe1')
    # 4 Build test data and calculate results
    # test_data = calc_error_from_meas_and_ref(df_meas, df_ref, 0)
    # test_results = TestResultZeroHz(test_name, amplitude_ua, test_data, 1000 / period1)
    # 5 Publish results
    # plot_data_frame(test_data, test_name, "Amplitude (uA/mV")
    plot_data_frame(dft, test_name, "Amplitude (uA/mV")
    # test_results.print_report()

    return df_raw_meas, dft  # test_data, test_results
"""


def test_zerohz(amplitude_ua, load_resistance, cycles=1, filter_window=20):
    """Test ZeroHz capability and publishes a html plot and html report
    :param load_resistance: Resistors between E1 and IE, E2 and IE
    :param filter_window: to reduce system noise
    :param amplitude_ua:
    :param cycles: when running chaining tests more than one cycle may be required
    :return: test_data, test_results
    """

    test_name = f'ZeroHz_12s_2_2_2_SPR_{amplitude_ua:.0f}uA_n42uAoff'
    signals = (iwe1, iwe2, iie, vwe1, vwe2)
    offset = -42
    sample_rate = 1000
    samples_cycle_est = int(12.12 * sample_rate)  # assume frequency is off by 10%
    samples = int(samples_cycle_est * (cycles + 1))  # 1 cycle added to assure capture of >1 contiguous cycle
    # 1 Get Waveform and filter
    df_raw_meas = get_iv_data(sample_rate, samples, amplitude_ua)
    df_raw_meas[:] = df_raw_meas[:].rolling(window=filter_window, min_periods=1, win_type='blackmanharris').mean()
    # Measure frequency measurement so reference can be created at same frequency
    period1 = measure_period(df_raw_meas, 'Iwe1', offset, cycles)
    period2 = measure_period(df_raw_meas, 'Iwe2', offset, cycles)
    samples_in_cycle = int((period1 + period2) / 2)
    # 2 Calculate Reference Waveform
    df_ref = build_zerohz_waveform_from_template(amplitude_ua=amplitude_ua, samples=samples_in_cycle,
                                                 load_resistance=load_resistance, offset_ua=offset)
    # 3 Extract period of measure based upon best correlation to reference (more reliable than edge or level triggering
    df_meas = extract_measurement_that_matches_ref(df_ref, df_raw_meas, 'Iwe1')
    # 4 Build test data and calculate results
    test_data = calc_error_from_meas_and_ref(df_meas, df_ref, 0)
    test_results = TestResultZeroHz(test_name, amplitude_ua, test_data, 1000 / period1)
    # 5 Publish results
    plot_data_frame(test_data, test_name, "Amplitude (uA/mV")
    test_results.print_report()

    return test_data, test_results


def measure_period(df, column, trigger_level=0, cycles=2):
    """Measure the Period of specified channel assuming the data does not produce more than 2 zero crossings per cycle
    :param df:
    :param column:
    :param trigger_level:
    :param cycles: used to determine whether too many zero crossings indicating noisy signal
    :return:
    """
    """Measure the frequency where cycles just validates the number of expected zero crossings"""
    b = np.array(df[column].values.tolist())  # convert to numpy array

    pos_edges = np.flatnonzero((b[:-1] > trigger_level) & (b[1:] < trigger_level)) + 1
    neg_edges = np.flatnonzero((b[:-1] < trigger_level) & (b[1:] > trigger_level)) + 1
    half_periods = np.diff(np.sort(np.concatenate((neg_edges, pos_edges), axis=0)))
    number_of_half_periods = len(half_periods)
    if number_of_half_periods > cycles * 3:
        print(f'Excessive number of 1/2 periods detected: {number_of_half_periods}, '
              f'signal may be too noisy for threshold may be set incorrectly')
    return half_periods[-2] + half_periods[-1]  # pick last two as the first could have signal disturbance


def rms_error_between_ref_and_shifted_measurement(df_ref, df_meas, column, shift):
    """
    Given a reference shift measurement and measure error for a particular column of data
    :param df_ref:
    :param df_meas:
    :param column:
    :param shift:
    :return: rms_error
    """

    shift = int(round(shift, 0))
    ref_len = df_ref.shape[0]
    rms_error = (df_ref[column].values - df_meas[column].iloc[shift:shift + ref_len].values).std()
    # df_meas = df_meas.shift(periods=shift, fill_value=0)
    return rms_error


def extract_measurement_that_matches_ref(df_ref, df_meas, column):
    """
    find the optimal alignment between to frames to minimize the rms error in col
    :param df_meas:
    :param df_ref:
    :param column:
    :return:  x: shift of df2 to get to the lowest error fun
    """
    from scipy import optimize

    def f(x):
        """Scan through measurement with ref and find minimum error"""
        nonlocal df_meas
        nonlocal df_ref
        nonlocal column
        return rms_error_between_ref_and_shifted_measurement(df_ref, df_meas, column, int(round(x, 0)))

    ref_len = df_ref.shape[0]
    res = optimize.minimize_scalar(f, bounds=(0, ref_len), method='bounded')
    # x is the shift and fun is the rms level
    start_loc = int(round(res.x, 0))
    return df_meas.iloc[start_loc:start_loc + ref_len]


def calc_error_from_meas_and_ref(df_meas, df_ref, shift):
    """ Combines the measured and calculated data frames aligned by a shift
    Calculates errors from df_meas
    Calculates expected result from df_calc
    :param df_meas:
    :param df_ref:
    :param shift:
    :return:
    """
    columns = ['Time', 'mIwe1', 'mIwe2', 'mIie', 'mVwe1', 'mVwe2', 'rIwe1', 'rIwe2', 'rIie', 'rVwe1', 'rVwe2',
               'eIwe1', 'eIwe2', 'eIie', 'eVwe1', 'eVwe2']
    df_ref = df_ref.shift(periods=shift, fill_value=0)
    df = pd.DataFrame(columns=columns)
    df.Time = df_ref.Time
    df.mIwe1 = df_meas.Iwe1.values
    df.mIwe2 = df_meas.Iwe2.values
    df.mIie = df_meas.Iie.values
    df.mVwe1 = df_meas.Vwe1_ie.values
    df.mVwe2 = df_meas.Vwe2_ie.values
    df.rIwe1 = df_ref.Iwe1.values
    df.rIwe2 = df_ref.Iwe2.values
    df.rIie = df_ref.Iie.values
    df.rVwe1 = df_ref.Vwe1.values
    df.rVwe2 = df_ref.Vwe2.values
    df.eIwe1 = df_meas.Iwe1.values - df_ref.Iwe1.values
    df.eIwe2 = df_meas.Iwe2.values - df_ref.Iwe2.values
    df.eIie = df_meas.Iie.values - df_ref.Iie.values
    df.eVwe1 = df_meas.Vwe1_ie.values - df_ref.Vwe1.values
    df.eVwe2 = df_meas.Vwe2_ie.values - df_ref.Vwe2.values
    return df


def _rms_error_between_signals_for_shift(df_meas, df_ref, column='Iwe1', shift=0):
    shift = int(round(shift, 0))
    df_ref = df_ref.shift(periods=shift, fill_value=0)
    return (df_meas[column].values - df_ref[column].values).std()


def _time_shift_to_minimize_error_between_frames(df_meas, df_ref, column='Iwe1', shift_range=200):
    """
    find the optimal alignment between to frames to minimize the rms error in col
    :param shift_range:
    :param df_meas:
    :param df_ref:
    :param column:
    :return:  x: shift of df2 to get to the lowest error fun
    """
    from scipy import optimize

    def f(x):
        nonlocal df_meas
        nonlocal df_ref
        nonlocal column
        shift = int(round(x, 0))  # only shift by integers
        return _rms_error_between_signals_for_shift(df_meas, df_ref, column=column, shift=shift)

    res = optimize.minimize_scalar(f, [-shift_range, shift_range])
    # x is the shift and fun is the rms level
    return int(round(res.x, 0)), res.fun


class TestResultAC:
    """From an expected amplitude, and frequency and a measurement/calculated frame create a test report"""

    def __init__(self, test_title, df, amplitude_ua, pw_us=60, rpw=8, ipi_us=30, dt1_us=510):
        from datetime import datetime
        self.id = test_title
        self.datetime = datetime.now()
        self.test_software = test_set_config.test_software
        self.test_software_build = test_set_config.test_software_build
        self.cpa_software = test_set_config.cpa_software
        self.cpa_software_build = test_set_config.test_software_build
        self.epg_sn = test_set_config.epg_sn
        self.daq_adapter = test_set_config.daq_adapter
        self.nidaq_product_type = daq.product_type
        self.nidaq_serial_number = daq.serial_number
        self.period = ['1', df.loc['Period', 'Iwe1-Meas'], df.loc['Period', 'Iwe2-Meas'],
                       *0.95, (pw_us * (1 + rpw) + ipi_us + dt1_us) * 1.05,'TECH10074.A13', 'Period', 'uS']
        self.pw_h = ['2', df.loc['PW-H', 'Iwe1-Meas'], df.loc['PW-H', 'Iwe2-Meas'],
                     pw_us * 0.95, pw_us * 1.05, 'TECH10074.A10', 'PW-H', 'uS']
        self.pw_l = ['3', df.loc['PW-L', 'Iwe1-Meas'], df.loc['PW-L', 'Iwe2-Meas'],
                     pw_us * 0.95, pw_us * 1.05, 'TECH10074.A10', 'PW-L', 'uS']
        self.pw_amp = ['4', df.loc['PW-AMP', 'Iwe1-Meas'], df.loc['PW-AMP', 'Iwe2-Meas'],
                       amplitude_ua * 0.95, amplitude_ua * 1.05, 'TECH10074.A9', 'PW-AMP', 'uA']
        self.ipi = ['5', df.loc['IPI', 'Iwe1-Meas'], df.loc['IPI', 'Iwe2-Meas'],
                    ipi_us * 0.95, ipi_us * 1.05, 'TECH10074.A11', 'IPI', 'uS']
        self.rt = ['6', df.loc['PW-RT', 'Iwe1-Meas'], df.loc['PW-RT', 'Iwe2-Meas'],
                   1, 10, 'TECH10074.A5', 'PW-RT', 'uS']
        self.ft = ['7', df.loc['PW-FT', 'Iwe1-Meas'], df.loc['PW-FT', 'Iwe2-Meas'],
                   1, 10, 'TECH10074.A5', 'PW-FT', 'uS']
        self.rpw_h = ['2', df.loc['RPW-H', 'Iwe1-Meas'], df.loc['RPW-H', 'Iwe2-Meas'],
                      pw_us * rpw * 0.95, pw_us * rpw * 1.05, 'TECH10074.A12', 'RPW-H', 'uS']
        self.rpw_l = ['3', df.loc['RPW-L', 'Iwe1-Meas'], df.loc['RPW-L', 'Iwe2-Meas'],
                      pw_us * rpw * 0.95, pw_us * rpw * 1.05, 'TECH10074.A12', 'RPW-L', 'uS'],
        self.rpw_amp = ['4', df.loc['RPW-AMP', 'Iwe1-Meas'], df.loc['RPW-AMP', 'Iwe2-Meas'],
                        (amplitude_ua / rpw) * 0.95, (amplitude_ua / rpw) * 1.05, 'TECH10074.A12', 'RPW-AMP', 'uA']

        self.results = self.create_dataframe()

    def create_dataframe(self):
        """Create a table of the results into a pandas frame so that it can be output in html"""
        results = pd.DataFrame(
            columns=['TestID', 'Parameter', 'Units', 'Measured', 'RangeA', 'RangeB', 'Pass/Fail', 'Requirement'])

        results = results.append(create_series(self.period), ignore_index=True)
        results = results.append(create_series(self.pw_h), ignore_index=True)
        results = results.append(create_series(self.pw_l), ignore_index=True)
        results = results.append(create_series(self.pw_amp), ignore_index=True)
        results = results.append(create_series(self.ipi), ignore_index=True)
        results = results.append(create_series(self.rt), ignore_index=True)
        results = results.append(create_series(self.ft), ignore_index=True)
        results = results.append(create_series(self.rpw_h), ignore_index=True)
        results = results.append(create_series(self.rpw_l), ignore_index=True)
        results = results.append(create_series(self.rpw_amp), ignore_index=True)
        return results

    def print_report(self):
        """Print HTML report"""
        f = open(self.id + '_report.html', 'w')
        lines = [
            f"<!DOCTYPE html>",
            f"<html><head><h2>Test Report</h2></head>",
            f"<body><p>Test ID: {self.id}</p>",
            f"<body><p>Files - Results :{self.id}_report.html,   Data File:_.csv,   Plot File:_.html</p>",
            f"<p>Reference Document: {test_document}</p>",
            f"<p>Date/Time: {self.datetime}</p>",
            f"<p>EPG SN: {test_set_config.epg_sn}</p>",
            f"<p>CPA: {test_set_config.cpa_software},  Build ID: {test_set_config.cpa_software_build}</p>",
            f"<p>DAQ Model #: {self.nidaq_product_type},  DAQ Tool #: {test_set_config.daq_tool},"
            f"  Adapter Tool #: {test_set_config.daq_adapter}</p>",
            f"<p>Test Set Software: {test_set_config.test_software},  "
            f"Build ID: {test_set_config.test_software_build}</p>",
            f"</body></html>"
        ]
        for line in lines:
            f.write(line)
        message = self.results.to_html()
        f.write(message)
        f.close()


class TestResultZeroHz:
    """From an expected amplitude, and frequency and a measurement/calculated frame create a test report"""

    def __init__(self, test_title, amplitude_ua, df, frequency):
        from datetime import datetime
        self.id = test_title
        self.datetime = datetime.now()
        self.test_software = test_set_config.test_software
        self.test_software_build = test_set_config.test_software_build
        self.cpa_software = test_set_config.cpa_software
        self.cpa_software_build = test_set_config.test_software_build
        self.epg_sn = test_set_config.epg_sn
        self.daq_adapter = test_set_config.daq_adapter
        self.nidaq_product_type = daq.product_type
        self.nidaq_serial_number = daq.serial_number
        self.peak_amp_ua = amplitude_ua
        self.rms_err_Vwe1 = df.eVwe1.std()
        self.rms_err_Vwe1 = df.eVwe1.std()
        self.vpp_err_Iwe1 = abs(df.eIwe1.max() - df.eIwe1.min())
        self.vpp_err_Iwe2 = abs(df.eIwe2.max() - df.eIwe2.min())
        self.vpp_err_Iie = abs(df.eIie.max() - df.eIie.min())
        self.vpp_err_Vwe1 = abs(df.eVwe1.max() - df.eVwe1.min())
        self.vpp_err_Vwe2 = abs(df.eVwe2.max() - df.eVwe2.min())
        self.iwe1_max = \
            ['1', df.mIwe1.max(), df.rIwe1.max() * 0.95, df.rIwe1.max() * 1.05, 'TECH10074.A1', 'Iwe1-max', 'uA']
        self.iwe1_min = \
            ['2', df.mIwe1.min(), df.rIwe1.min() * 0.95, df.rIwe1.min() * 1.05, 'TECH10074.A1', 'Iwe1-min', 'uA']
        self.rms_err_Iwe1 = \
            ['3', df.eIwe1.std(), -10, 10, 'TECH10074.A10', 'Iwe1-rms-error', 'uA']
        self.iwe2_max = \
            ['4', df.mIwe2.max(), df.rIwe2.max() * 0.95, df.rIwe2.max() * 1.05, 'TECH10074.A1', 'Iwe2-max', 'uA']
        self.iwe2_min = \
            ['5', df.mIwe2.min(), df.rIwe2.min() * 0.95, df.rIwe2.min() * 1.05, 'TECH10074.A1', 'Iwe2-min', 'uA']
        self.rms_err_Iwe2 = \
            ['6', df.eIwe2.std(), -10, +10, 'TECH10074.A10', 'Iwe1-rms-error', 'uA']
        self.iie_max = ['7', df.mIie.max(), df.rIie.max() - 10, df.rIie.max() + 10, 'TECH10074.A1', 'Iie-max', 'uA']
        self.iie_min = ['8', df.mIie.min(), df.rIie.min() - 10, df.rIie.min() + 10, 'TECH10074.A1', 'Iie-min', 'uA']
        self.rms_err_Iie = \
            ['9', df.eIie.std(), -10.0, +10.0, 'TECH10074.A10', 'Iwe1-rms-error', 'uA']
        self.vwe1_max = \
            ['10', df.mVwe1.max(), df.rVwe1.max() * 0.95, df.rVwe1.max() * 1.05, 'TECH10074.A6', 'Vwe1-max', 'mV']
        self.vwe1_min = \
            ['11', df.mVwe1.min(), df.rVwe1.min() * 0.95, df.rVwe1.min() * 1.05, 'TECH10074.A6', 'Vwe1-min', 'mV']
        self.vwe2_max = \
            ['12', df.mVwe2.max(), df.rVwe2.max() * 0.95, df.rVwe2.max() * 1.05, 'TECH10074.A6', 'Vwe1-max', 'mV']
        self.vwe2_min = \
            ['13', df.mVwe2.min(), df.rVwe2.min() * 0.95, df.rVwe2.min() * 1.05, 'TECH10074.A6', 'Vwe1-min', 'mV']
        self.frequency = \
            ['14', frequency * 1000, 1000 / (12 * 0.99), 1000 / (12 * 1.01), 'TECH10074.A9', 'Frequency', 'mHz']
        self.results = self.create_dataframe()

    def create_dataframe(self):
        """Create a table of the results into a pandas frame so that it can be output in html"""
        results = pd.DataFrame(
            columns=['TestID', 'Parameter', 'Units', 'Measured', 'RangeA', 'RangeB', 'Pass/Fail', 'Requirement'])

        results = results.append(create_series(self.iwe1_max), ignore_index=True)
        results = results.append(create_series(self.iwe1_min), ignore_index=True)
        results = results.append(create_series(self.rms_err_Iwe1), ignore_index=True)
        results = results.append(create_series(self.iwe2_max), ignore_index=True)
        results = results.append(create_series(self.iwe2_min), ignore_index=True)
        results = results.append(create_series(self.rms_err_Iwe2), ignore_index=True)
        results = results.append(create_series(self.iie_max), ignore_index=True)
        results = results.append(create_series(self.iie_min), ignore_index=True)
        results = results.append(create_series(self.rms_err_Iie), ignore_index=True)
        results = results.append(create_series(self.vwe1_max), ignore_index=True)
        results = results.append(create_series(self.vwe1_min), ignore_index=True)
        results = results.append(create_series(self.vwe1_max), ignore_index=True)
        results = results.append(create_series(self.vwe2_min), ignore_index=True)
        results = results.append(create_series(self.frequency), ignore_index=True)
        return results

    def print_report(self):
        """Print HTML report"""
        f = open(self.id + '_report.html', 'w')
        lines = [
            f"<!DOCTYPE html>",
            f"<html><head><h2>Test Report</h2></head>",
            f"<body><p>Test ID: {self.id}</p>",
            f"<body><p>Files - Results :{self.id}_report.html,   Data File:_.csv,   Plot File:_.html</p>",
            f"<p>Reference Document: {test_document}</p>",
            f"<p>Date/Time: {self.datetime}</p>",
            f"<p>EPG SN: {test_set_config.epg_sn}</p>",
            f"<p>CPA: {test_set_config.cpa_software},  Build ID: {test_set_config.cpa_software_build}</p>",
            f"<p>DAQ Model #: {self.nidaq_product_type},  DAQ Tool #: {test_set_config.daq_tool},"
            f"  Adapter Tool #: {test_set_config.daq_adapter}</p>",
            f"<p>Test Set Software: {test_set_config.test_software},  "
            f"Build ID: {test_set_config.test_software_build}</p>",
            f"</body></html>"
        ]
        for line in lines:
            f.write(line)
        message = self.results.to_html()
        f.write(message)
        f.close()


def create_series(rl):
    """Given measurement and expected range create a row in frame
    :param rl: [Test ID, Measured, Range A, Range B, Requirement Trace, Signal, Unit]
    """
    result = 'Pass' if (rl[2] <= rl[1] <= rl[3]) | (rl[2] >= rl[1] >= rl[3]) else 'Fail'  # in between
    results = pd.Series({'TestID': rl[0], 'Parameter': rl[5], 'Units': rl[6],
                         'Measured': round(rl[1], 1), 'RangeA': round(rl[2], 1), 'RangeB': round(rl[3], 1),
                         'Pass/Fail': result, 'Requirement': rl[4]}, )
    return results


def _tsm_test_script(test_script="Irving Stimulation Test Scenarios V0.xlsx"):
    """
    Prompts tester to set up DUT under sequence specified by spreadsheet and fills in test results
    Displays and records each test
    :return: dm dataframe with results
    """
    import winsound

    results_file = data_path() + 'results.csv'
    try:
        results_file_status = open(results_file, "w")
        results_file_status.close()
    except IOError:
        print(f'{results_file} file open, please close')
        winsound.Beep(200, 250)
        return

    max_nidaq_sample_rate = int(daq.max_multi_chan_rate / len(signals))

    dm = pd.read_excel(data_path() + test_script, header=1, nrows=14)
    for index, row in dm.iterrows():
        if "ZeroHz" in row["Name"]:
            plot_title = f'{row["ID"]}_{row["Name"]}_Load_{row["Load"]}_Amp_{1000 * row["Amp (mA)"]:.0f}uA_' \
                         f'Bias_{row["Bias (uA)"]:.0f}uA'
            plot_title = plot_title.replace(" ", "_").replace(".", "p").replace("-", "n")
            filter_window = 50
            sample_rate = 10000
            cycles = 1
        else:
            plot_title = f'{row["ID"]}_{row["Name"]}_Load_{row["Load"]}_Amp_{1000 * row["Amp (mA)"]:.0f}uA_' \
                         f'Freq_{row["Frequency (Hz)"]:.0f}Hz_PW_{row["PW (uS)"]:.0f}uS_' \
                         f'DT_{row["RPR"]:.0f}_{row["IPI (uS)"]:.0f}'
            plot_title = plot_title.replace(" ", "_").replace(".", "p").replace("-", "n")
            filter_window = 1
            cycles = 5
            sample_rate = min(int(100e6 / row["PW (uS)"]), max_nidaq_sample_rate)
        print(plot_title)
        answer = input("Start Stm on Device then (cr) to test or (s)kip or (q)uit").lower()
        result = 'repeat'
        if 'q' in answer:
            dm.to_csv(data_path() + 'results.csv')
            print(f'Partial data Saved to {data_path()}results.csv')
            break
        elif 's' in answer:
            continue
        else:
            while 'r' in result:  # will loop until use indicates done
                print(f'Capturing data...', end=" ")
                maximum_current_ma = row["Amp (mA)"]
                samples_cycle = int((1 / row["Frequency (Hz)"]) * sample_rate)
                df, dft, frequency, pulsewidth = \
                    _stimulation_cycle_test(plot_title=plot_title, trigger_channel='Iwe2',
                                            maximum_current_ma=maximum_current_ma,
                                            sample_rate=sample_rate, samples_cycle=samples_cycle, cycles=cycles,
                                            window=filter_window)
                channels = ['Iwe1', 'Iwe2', 'Iie']
                for channel in channels:
                    dm.loc[index, channel + '_max'] = df[channel].max()
                    dm.loc[index, channel + '_min'] = df[channel].min()
                    dm.loc[index, channel + '_DC'] = df[channel].mean()
                dm.loc[index, 'Freq_Hz'] = frequency
                dm.loc[index, 'PW_uS'] = pulsewidth
                dm.loc[index, 'Bat_mA'] = df['Bat_mA'].mean()
                dm.loc[index, 'Vwe1span'] = abs(df['Vwe1_ie'].max() - df['Vwe1_ie'].min())
                dm.loc[index, 'Vwe2span'] = abs(df['Vwe2_ie'].max() - df['Vwe2_ie'].min())
                dm.loc[index, "TstSetRes_uA"] = \
                    get_device_accuracy(get_minimum_current_range(row["Amp (mA)"]), "CURRENT")
                winsound.Beep(500, 100)
                result = input("repeat(r) or (cr) to continue")
    dm.to_csv(results_file)
    print(f'Data Saved to {results_file}')
    return


def _stimulation_cycle_test(plot_title, trigger_channel='Iwe1', maximum_current_ma=5, sample_rate=100,
                            samples_cycle=1200, cycles=1, trigger_level_percent=5, window=1):
    """
    Stimulation Cycle Test is designed to test stimulation with the fewest parameters
    :param window:
    :param plot_title: is used to description the graph and tag the results
    :param maximum_current_ma: is used to set the gain with maximum resolution without clipping
    :param sample_rate:
    :param samples_cycle:
    :param cycles
    :param trigger_channel can be an index or description
    :param trigger_level_percent: Units in percent
    :return: df, frequency, pulsewidth
    """
    samples = int(samples_cycle * (cycles + 1))  # 1 cycle added for triggering
    df = get_iv_data(sample_rate, samples, maximum_current_ma)
    capture_boundary = int(samples_cycle / 200)  # show 1% of waveform before and after waveform
    df[:] = df[:].rolling(window=window, min_periods=1, win_type='blackmanharris').mean()
    dft, frequency, pulsewidth = \
        _software_trigger(df, trigger_channel, sample_rate,
                          trigger_level=-int(trigger_level_percent * maximum_current_ma * 1000 / 100),
                          post_trigger_samples=samples_cycle * cycles + 2 * capture_boundary,
                          pre_trigger_samples=capture_boundary, negative_edge_trigger=True)

    plot_data_frame_left_right(dft, plot_title,
                               'Current (uA)', [i.Name for i in signals if i.Type == 'current'],
                               'Voltage(mV)', [i.Name for i in signals if i.Type == 'voltage'])
    return df, dft, frequency, pulsewidth


def get_iv_data(sample_rate, samples, maximum_current_ua):
    """
    Get current, voltage data from first Nidaq board in list
    :param maximum_current_ua:  used to maximize current range
    :param sample_rate:
    :param samples:
    :return: df frame
    """

    # Define physical acquisition signals i.e. 'Dev1/ai0, Dev1/ai2'
    current_channels = ', '.join([daq.name + '/' + i.Chan for i in signals if i.Type == 'current'])
    voltage_channels = ', '.join([daq.name + '/' + i.Chan for i in signals if i.Type == 'voltage'])
    current_range = get_minimum_current_range(maximum_current_ua)
    voltage_range = 10.0  # voltage resolution is maxed because voltages are not critical

    with nidaqmx.Task() as ai_task:
        if len(current_channels) > 0:
            ai_task.ai_channels.add_ai_voltage_chan(current_channels,
                                                    max_val=current_range, min_val=-current_range,
                                                    terminal_config=TerminalConfiguration.DIFFERENTIAL)
        if len(voltage_channels) > 0:
            ai_task.ai_channels.add_ai_voltage_chan(voltage_channels,
                                                    max_val=voltage_range, min_val=-voltage_range,
                                                    terminal_config=TerminalConfiguration.DIFFERENTIAL)
        ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples, sample_mode=AcquisitionType.FINITE)
        ai_task.start()
        da = np.array(ai_task.read(number_of_samples_per_channel=samples,
                                   timeout=samples / sample_rate + 1 + 100)).transpose()
    # transform voltages to currents
    multiplier = [i.Mult for i in signals]
    dt = np.linspace(0, samples / sample_rate, samples)
    da = da * np.array(multiplier)
    columns = ['Time'] + list([i.Name for i in signals])
    da = np.column_stack((dt, da))
    df = pd.DataFrame(data=da[0:, 0:], columns=columns)

    return df


def get_single_channel(channel='ai0', voltage_range=10.0, sample_rate=1e6, samples=50000,
                       triggered=False, trigger_level=1, pre_trigger_samples=0):
    timeout = samples / sample_rate + 1 + 100
    physical_channel = daq.name + '/' + channel

    with nidaqmx.Task() as ai_task:
        ai_task.ai_channels.add_ai_voltage_chan(physical_channel,
                                                max_val=voltage_range, min_val=-voltage_range,
                                                terminal_config=TerminalConfiguration.DIFFERENTIAL)
        ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples,
                                           sample_mode=AcquisitionType.FINITE)
        if triggered:
            ai_task.control(action=TaskMode.TASK_VERIFY)
            if trigger_level >= 0:
                ai_task.triggers.reference_trigger.cfg_anlg_edge_ref_trig(trigger_source=physical_channel,
                                                                          pretrigger_samples=pre_trigger_samples,
                                                                          trigger_slope=Slope.RISING,
                                                                          trigger_level=trigger_level)
            else:
                ai_task.triggers.reference_trigger.cfg_anlg_edge_ref_trig(trigger_source=physical_channel,
                                                                          pretrigger_samples=pre_trigger_samples,
                                                                          trigger_slope=Slope.FALLING,
                                                                          trigger_level=trigger_level)
        else:
            ai_task.start()

        da = np.array(ai_task.read(number_of_samples_per_channel=samples, timeout=timeout))
    return da


def get_minimum_current_range(max_amplitude_ua, sense_resistor=50):
    """ Given maximum current (amplitude) and sense resistor determine best voltage range for current sensors
    :param sense_resistor: is assumed to be 47 ohms
    :param max_amplitude_ua: maximum current expected
    :return is the lowest possible range from nidaq device
    """
    max_voltage = max_amplitude_ua * sense_resistor / 1e6  # 10% margin
    a = np.abs(np.array(daq.ai_voltage_rngs))
    return a[np.argmax(a > max_voltage)]


def get_minimum_voltage_range(max_voltage):
    """ Given maximum current (amplitude) and sense resistor determine best voltage range for current sensors

    :return is the lowest possible range from nidaq device
    """
    a = np.abs(np.array(daq.ai_voltage_rngs))
    return a[np.argmax(a > max_voltage)]


def get_device_accuracy(voltage_range, measure="voltage", resistor_value=47):
    """Gets accuracy of first NI device connected in list
    'voltage_range' is the range that is used in the device. 20 not implemented with divider yet!
    'measure' is either 'VOLTAGE' or 'CURRENT' where current needs a sense resistor
    'resistor_value' depends on test set sense resistor
    """
    product_type = daq.product_type
    df = None
    if "6361" in product_type:
        df = pd.DataFrame(np.array([[10, 315, 1660],
                                    [5, 157, 870],
                                    [2, 64, 350],
                                    [1, 38, 190],
                                    [0.5, 27, 100],
                                    [0.2, 21, 53],
                                    [0.1, 17, 33]]),
                          columns=['Range', 'Noise', 'Accuracy'])
    elif "6251" in product_type:
        df = pd.DataFrame(np.array([[10, 280, 1920],
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


def plot_fft(df, chart_title="Default"):
    from bokeh.plotting import figure, output_file, show, ColumnDataSource
    from bokeh.models import Label, Column, Range1d
    import scipy.fftpack
    from scipy.signal.windows import blackman

    # FFT
    samples = df.shape[0]
    sample_rate = round(1 / (df['Time'].loc[1] - df['Time'].iloc[0]))
    period = 1 / sample_rate
    xf = np.linspace(0.0, 1.0 / (2.0 * period), samples // 2)
    yf = scipy.fft(df['?'])

    w = blackman(samples)
    ywf = scipy.fft(df['?'] * w)
    p_rec = 2.0 / samples * np.abs(yf[0:samples // 2])
    p_bm = 2.0 / samples * np.abs(ywf[0:samples // 2])

    data = ColumnDataSource(dict(df))

    width = 1200
    height = 300

    text = chart_title + f': Sample Rate: {sample_rate:.0f}Hz, 47ohm SenseR, 1uA/53uV Resolution'

    output_file(data_path() + chart_title + '.html')

    p = figure(title=text, x_axis_label='Time(s)',
               y_axis_label='Voltage/Current (mV/uA)', plot_width=width, plot_height=height)

    p.line(x='Time', y='Iie', source=data, legend_label='Iie', line_color='blue')

    text = chart_title + f': Sample Rate: {sample_rate:.0f}Hz, FFT - 1MHz SR + Window '
    s = figure(title=text, x_axis_label='Frequency (Hz)', y_axis_label='Power',
               plot_width=width, plot_height=height, y_axis_type="log")
    s.line(x=xf, y=p_rec, line_color='red', legend_label='Rectangular window')
    s.line(x=xf, y=p_bm, line_color='green', legend_label='Blackman window')
    s.y_range = Range1d(start=0.001, end=100)

    mean = df['Iie'].mean()
    stddev = df['Iie'].std()
    peak_peak = np.abs(df['Iie'].max() - df['Iie'].min())
    text1 = f'Mean:{mean:.2f} AC RMS(std): {stddev:.2f} Vpp: {peak_peak:.2f}uA'

    citation1 = Label(x=70, y=70, x_units='screen', y_units='screen',
                      text=text1, render_mode='css',
                      border_line_color='black', border_line_alpha=1.0,
                      background_fill_color='white', background_fill_alpha=1.0)

    p.add_layout(citation1)
    show(Column(p, s))


def _software_trigger(df, signal, sample_rate, pre_trigger_samples=0, post_trigger_samples=100,
                      trigger_level=-100, negative_edge_trigger=True):
    """Take waveform, find trigger and trim of in front of trigger
    :param df:
    :param signal: column in frame either integer or string
    :param sample_rate: used to compute frequency and pulse width
    :param pre_trigger_samples: samples to keep before trigger
    :param post_trigger_samples: samples to keep after trigger
    :param trigger_level:
    :param negative_edge_trigger: boolean, either negative or positive edge trigger
    :return df:
    :return frequency: 1/({-}trig[1] - {-}trig[0]) either neg or pos trigger
    :return pulsewidth: 1/({+T}trig[0] - {-}trig[0]) either neg or pos trigger

    Strips off data before first negative slope zero crossing on channel Iwe1
    - add capability to keep remaining number of cycles
    """

    b = np.array(df[signal].values.tolist())  # convert to numpy array

    neg_edges = np.flatnonzero((b[:-1] > trigger_level) & (b[1:] < trigger_level)) + 1
    pos_edges = np.flatnonzero((b[:-1] < trigger_level) & (b[1:] > trigger_level)) + 1
    print(f'trigger level:{trigger_level}  Pos: {pos_edges}  Neg: {neg_edges}')

    #   get the first edge that has a positive index
    pre_edge = 0  # by default no trigger
    for edge in neg_edges:
        if (edge - pre_trigger_samples) >= 0:
            pre_edge = edge - pre_trigger_samples
            neg_edges = neg_edges - pre_trigger_samples  # for external reporting
            pos_edges = pos_edges - pre_trigger_samples
            break

    # trim samples desired capture range
    trigger_time = pre_edge / sample_rate  # convert to time
    stop_time = trigger_time + post_trigger_samples / sample_rate
    df = df.truncate(before=trigger_time, after=stop_time)  # uses index value not index number or sample

    # Set start time to zero
    start_time = df.index.values[0]
    df.index = df.index - start_time

    if len(neg_edges) > 1:
        samples_between_trigger = neg_edges[1] - neg_edges[0]
    else:
        samples_between_trigger = 1
    frequency = sample_rate / samples_between_trigger

    period_us = 1e6 / sample_rate

    if (len(neg_edges) > 1) & (len(pos_edges) > 1):
        pulsewidth = np.abs((pos_edges[0] - neg_edges[0]) * period_us)
        print(f'Trig Info sr:{sample_rate} pw:{pulsewidth:.0f} freq:{frequency:.0f} pos0:{pos_edges[0]}'
              f' neg0:{neg_edges[0]} pos1:{pos_edges[1]} neg1:{neg_edges[1]}')
    else:
        pulsewidth = 0

    return df, frequency, pulsewidth


def add_accumulated_charge_to_data_frame(df):
    """Add Columns of running charge in uC for all current measuring channels
    :param df:
    :return:
    """
    charge_columns = []  # hold charge column names
    sample_period = (df.index[-1] - df.index[0]) / len(df)
    list_current_signals = [i.Name for i in signals if i.Type == 'current']
    for col in list_current_signals:
        new_col = col + 'Chg'
        charge_columns = charge_columns + [new_col]
        df[new_col] = df[col] * sample_period  # seconds * uA = uC
        df[new_col] = df[new_col].cumsum()

    return df


def plot_data_frame(df, plot_title='Test_plot', left_axis_label='Amplitude (uA/mV)'):
    """
    Plots a dataframe with specified columns on left and right axis respectfully
    :param left_axis_label:
    :param df:
    :param plot_title:
    """
    from bokeh.plotting import figure, output_file, show
    from bokeh.palettes import Category20

    if df.shape[1] > 2:
        colors = Category20[df.shape[1]]  # limited to 20 colors
    else:
        colors = ['#1f77b4', '#aec7e8']  # why do paletes have a min of 3?

    output_file(f'{data_path()}{plot_title}.html')

    p = figure(title=plot_title, x_axis_label='Time(s)', y_axis_label=left_axis_label,
               x_range=(df.Time[0], df.Time.iloc[-1] * 1.3))
    for i, col in enumerate(df.columns):
        if col != 'Time':  # Dont plot time
            p.line(x='Time', y=col, source=df, legend_label=col, color=colors[i])

    p.legend.click_policy = "hide"
    show(p)


def plot_data_frames(dr, df):
    from bokeh.plotting import figure, output_file, show
    output_file("temp.html")
    p = figure()
    p.line(x=df.Time, y=df.Iwe1, color='blue')
    p.line(x=dr.Time, y=dr.Iwe1, color='red', line_dash='dashed')
    show(p)


def plot_data_frame_left_right(df, plot_title, left_axis_label, left_columns, right_axis_label, right_columns):
    """
    Plots a dataframe with specified columns on left and right axis respectfully
    :param df:
    :param right_columns:
    :param left_columns:
    :param plot_title:
    :param left_axis_label:
    :param right_axis_label:
    :return:
    """
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import LinearAxis, Range1d
    from bokeh.palettes import Category10, Accent

    # plot currents on left axis
    output_file(f'{data_path()}{plot_title}.html')
    p = figure(title=plot_title, x_axis_label='Time(s)', y_axis_label=left_axis_label)
    left_max = -1000
    left_min = +1000
    c = max(3, len(left_columns))
    for i, col in enumerate(left_columns):
        p.line(x='index', y=col, source=df, color=Category10[c][i],
               legend_label=f'{col}:{df[col].mean():0.2f}')
        left_max = max(left_max, df[col].max() + 100)
        left_min = min(left_min, df[col].min() - 100)

    max_val = max(np.abs(left_max), np.abs(left_min))
    left_max = max_val
    left_min = -max_val

    p.y_range = Range1d(start=left_min, end=left_max)
    right_max = -1000
    right_min = +1000
    for col in right_columns:
        right_max = max(right_max, df[col].max())
        right_min = min(right_min, df[col].min())
    max_val = max(np.abs(right_max), np.abs(right_min))
    right_max = max_val
    right_min = -max_val
    p.extra_y_ranges = {"RA": Range1d(start=right_min, end=right_max)}
    p.add_layout(LinearAxis(y_range_name="RA", axis_label=right_axis_label), 'right')

    c = max(3, len(right_columns))
    for i, col in enumerate(right_columns):
        p.line(x='index', y=col, source=df, y_range_name="RA", legend_label=col, color=Accent[c][i])

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    show(p)


def plot_numpy_array(a):
    """
    quick plot of numpy array
    """
    from bokeh.plotting import figure, output_file, show
    output_file(data_path() + 'SimpleNumpyPlot.html')
    p = figure()
    x = np.linspace(0, len(a) - 1, len(a))
    p.line(x, a)
    show(p)


def read_irving_csv(path_filename):
    """
    Read CSV file produced by Irving and translate to data frame CPA_3.6 and later
    :param path_filename: path and file name
    """
    csv_hdr = pd.read_csv(path_filename, sep=',', header=None, nrows=8)
    samples = int(csv_hdr[1][6])
    stop = samples * int(csv_hdr[1][5]) / 1000.
    dt = np.linspace(0, stop, samples)
    df = pd.read_csv(path_filename, header=9)
    df = df.assign(Time=dt)
    return df


def build_zerohz_waveform_from_template(amplitude_ua=1000, offset_ua=-42, samples=12000, load_resistance=2000,
                                        template_file='12s(2-2-2)-SPRe.csv', template_duration=12,
                                        template_sample_rate=100):
    """Creates ZeroHz waveform from csv template and returns dataframe of samples requested
    :param amplitude_ua: scale amplitude in uA units
    :param offset_ua: in
    :param samples: samples to generate where sample rate of reference where output
                    sample rate = samples / template_duration
    :param load_resistance: only handles resistive loads
    :param template_file: CSV file format one column of data only
    :param template_duration: non-biased waveform where abs() is <= 1
    :param template_sample_rate: rate at which CSV is sampled since it has no time
    :return: generated data frame with the following columns ['Time', 'Iwe1', 'Iwe2', 'Iie', 'Vwe1', 'Vwe2']
    """
    from scipy import signal

    dr = pd.read_csv(template_file, header=None)
    template_samples = dr.shape[0]

    # Build reference frame at its native sample rate
    dr['Time'] = np.linspace(0, template_duration - 1 / template_sample_rate, template_samples)
    dr['Iwe1'] = dr[0] * amplitude_ua + (offset_ua / 2)  # dr[0] raw data
    dr['Iwe2'] = -dr[0] * amplitude_ua + (offset_ua / 2)
    dr['Iie'] = offset_ua
    dr = dr.drop([0], axis=1)
    dr = dr.reset_index(drop=True)

    # build data at new sample rate
    new_time = np.linspace(0, template_duration - (1 / samples), samples)
    new_iwe1 = signal.resample(dr.Iwe1, samples)
    new_iwe2 = signal.resample(dr.Iwe2, samples)
    # add voltages with assumed resistive load
    new_vwe1 = new_iwe1 * load_resistance / 1000.0
    new_vwe2 = new_iwe2 * load_resistance / 1000.0
    # build pandas data frame
    df = pd.DataFrame(columns=['Time', 'Iwe1', 'Iwe2', 'Iie', 'Vwe1', 'Vwe2'])
    df['Time'] = new_time
    df['Iwe1'] = new_iwe1
    df['Iwe2'] = new_iwe2
    df['Iie'] = offset_ua
    df['Vwe1'] = new_vwe1
    df['Vwe2'] = new_vwe2

    return df


def set_digital_output(line=0, state=False):
    digital_output = f'{daq.name}/port0/line{line}'
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(digital_output)
        task.write(state, auto_start=True)
        task.stop()


def set_output_voltage(volts=1.0):
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(f'{daq.name}/ao0')
        task.write(volts, auto_start=True)
        task.stop()
