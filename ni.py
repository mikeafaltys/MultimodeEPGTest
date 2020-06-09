"""
Instructions:
    use requirements.txt to install all required python dependencies
    Manual Set Path below
    Use simple_capture_plot() for basic operations
"""

import os
import sys
import numpy as np
import pandas as pd
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.constants import Slope, TaskMode
import nidaqmx.system as system
from nidaqmx.constants import (LineGrouping)
from dataclasses import dataclass
import time


def data_path():
    return os.getcwd() + "\\Data\\"


class TestSet:
    """ Encapsulates test set functionality and configuration
    """
    def __init__(self, idx):
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
        # Test Set Config
        self.num_channels = 6
        self.max_sample_rate_per_channel = int(self.max_single_chan_rate / self.num_channels)
        self.sample_rate_per_channel = self.max_sample_rate_per_channel  # set to max sample rate unless overridden
        self.signals = ['Iwe1', 'Iwe2', 'Iie', 'Vwe1_ie', 'Vwe2_ie', 'Vie_0']
        self.physical_channels = ['ai0', 'ai1', 'ai2', 'ai8', 'ai9', 'ai10']
        self.physical_channel_string = ', '.join([self.name + '/' + i for i in self.physical_channels])
        self.physical_channel_by_signal = dict(zip(self.signals, self.physical_channel_string.split(',')))
        self.v_offset = [0, 0, 0, 0, 0, 0]
        self.sense_resistors = [10, 10, -49.9, 1000, 1000, 1000]  # 1000 is a
        self.amp_gains = [60.1716, 60.1716, 79.8644, 1, 1, 1]
        self.v_to_uamv = [1e6 / (a * b) for a, b in zip(self.amp_gains, self.sense_resistors)]
        self.uamv_to_v = [(a * b) for a, b in zip(self.amp_gains, self.sense_resistors)]
        self.v_offset_by_signal = dict(zip(self.signals, self.v_offset))
        self.v_to_uamv_by_signal = dict(zip(self.signals, self.v_to_uamv))
        self.uamv_to_v_by_signal = dict(zip(self.signals, self.uamv_to_v))
        self.amp_gain_by_signal = dict(zip(self.signals, self.amp_gains))
        self._load_resistance = 100
        print(f'Signals: {self.signals} being measured')

    @property
    def load_resistance(self):
        return self._load_resistance

    @load_resistance.setter
    def load_resistance(self, resistance):
        """Set Load Resistance
         :param self:
         :param resistance: 0-open, 100, 500, 1000, 10000, 'Calibrate'
         """
        if resistance == 100:
            self.set_digital_port0(int('00000011', 2))
        if resistance == 500:
            self.set_digital_port0(int('00001100', 2))
        if resistance == 1000:
            self.set_digital_port0(int('00110000', 2))
        if resistance == 10000:
            self.set_digital_port0(int('11000000', 2))
        if resistance == 'Calibration':
            self.set_digital_port0(int('00000010', 2))

    def set_digital_port0(self, lines):
        digital_output = f'{self.name}/port0/line0:7'
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(digital_output, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
            try:
                task.write(lines, auto_start=True)
                # task.stop()
            except nidaqmx.DaqError as e:
                print(e)

    def calibrate_daq_adapter_offset(self):
        """Offset Measured when there are no inputs presents
        :return:
        """
        self.load_resistance = 100
        self.power_on()
        sample_rate = 100000
        samples = 20000
        voltage_range = 10
        with nidaqmx.Task() as ai_task:
            try:
                ai_task.ai_channels.add_ai_voltage_chan(physical_channel=self.physical_channel_string,
                                                        max_val=voltage_range, min_val=-voltage_range,
                                                        terminal_config=TerminalConfiguration.RSE)
                ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples,
                                                   sample_mode=AcquisitionType.FINITE)
                ai_task.start()
                da = np.array(ai_task.read(number_of_samples_per_channel=samples,
                                           timeout=samples / sample_rate + 5)).transpose()
            except nidaqmx.DaqError as e:
                print(e)
                sys.exit(1)
        self.power_off()
        self.v_offset = np.mean(da, 0)
        print("Offset Calibration Measures: ", end=" ")
        over_expected_value = 0
        for signal, gain, sense, offset in zip(self.signals, self.amp_gains, self.sense_resistors, self.v_offset):
            if 'V' in signal:
                v_error = 1000 * offset / gain
                if abs(v_error) > 1:
                    over_expected_value += 1
                print(f'{signal}:{v_error:.3f}mV ', end=" ")
            else:
                i_error = 1e6 * offset / gain / sense
                if abs(i_error) > 1:
                    over_expected_value += 1
                print(f'{signal}:{i_error:.3f}uA ', end=" ")
        print('.')
        if over_expected_value > 0:
            print(f'Error greater than 1uA or 1mV, stop stimulation and/or check connections')

    def power_on(self):
        digital_output = f'{self.name}/port1/line0'
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(digital_output)
            task.write(True, auto_start=True)
        time.sleep(0.1)  # Wait 100mS for power to stabilize

    def power_off(self):
        digital_output = f'{self.name}/port1/line0'
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(digital_output)
            task.write(False, auto_start=True)

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
        df.set_index("Range", inplace=True)
        accuracy = df.loc[voltage_range, 'Accuracy']
        if measure == 'current':
            accuracy /= resistor_value
        return round(accuracy, 2)

    def get_iv_data(self, sample_rate, samples):
        """Get current, voltage data from first Nidaq board in list
        :param sample_rate:
        :param samples:
        :return: df frame
        """
        voltage_range = 10.0  # voltage resolution is maxed because voltages are not critical
        self.power_on()
        with nidaqmx.Task() as ai_task:
            try:
                ai_task.ai_channels.add_ai_voltage_chan(self.physical_channel_string,
                                                        max_val=voltage_range, min_val=-voltage_range,
                                                        terminal_config=TerminalConfiguration.RSE)
                ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples,
                                                   sample_mode=AcquisitionType.FINITE)
                ai_task.start()
                da = np.array(ai_task.read(number_of_samples_per_channel=samples,
                                           timeout=samples / sample_rate + 5)).transpose()
            except nidaqmx.DaqWarning as e:
                print('DaqWarning caught as exception: {0}\n'.format(e))
                assert e.error_code == 200015
                sys.exit(1)
        self.power_off()
        # transform voltages to currents
        dt = np.linspace(0, samples / sample_rate, samples)
        da = da - self.v_offset
        da = da * np.array(self.v_to_uamv)  # C
        columns = ['Time'] + self.signals
        da = np.column_stack((dt, da))
        df = pd.DataFrame(data=da[0:, 0:], columns=columns)
        return df

    def get_single_channel(self, signal, voltage_range=10.0, sample_rate=1e6, samples=50000,
                           triggered=False, trigger_level=1, pre_trigger_samples=0):
        physical_channel = self.physical_channel_by_signal[signal]
        self.power_on()
        timeout = samples / sample_rate + 5
        with nidaqmx.Task() as ai_task:
            try:
                ai_task.ai_channels.add_ai_voltage_chan(physical_channel,
                                                        max_val=voltage_range, min_val=-voltage_range,
                                                        terminal_config=TerminalConfiguration.RSE)
                ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples,
                                                   sample_mode=AcquisitionType.FINITE)
                if triggered:
                    ai_task.control(action=TaskMode.TASK_VERIFY)
                    if trigger_level >= 0:
                        ai_task.triggers.reference_trigger. \
                            cfg_anlg_edge_ref_trig(trigger_source=physical_channel,
                                                   pretrigger_samples=pre_trigger_samples,
                                                   trigger_slope=Slope.RISING, trigger_level=trigger_level)
                    else:
                        ai_task.triggers.reference_trigger. \
                            cfg_anlg_edge_ref_trig(trigger_source=physical_channel,
                                                   pretrigger_samples=pre_trigger_samples,
                                                   trigger_slope=Slope.FALLING, trigger_level=trigger_level)
                else:
                    ai_task.start()
                print(f'{signal} Range:{voltage_range} Trigger:{trigger_level}')
                da = np.array(ai_task.read(number_of_samples_per_channel=samples, timeout=timeout))
                da = da - self.v_offset_by_signal[signal]
            except nidaqmx.DaqWarning as e:
                print('DaqWarning caught as exception: {0}\n'.format(e))
                assert e.error_code == 200015
                sys.exit(1)
        self.power_off()
        return da

    def get_minimum_voltage_range(self, max_voltage):
        """ Given maximum expected voltage determine lowest feasible voltage range
        :return is the lowest possible range from nidaq device
        """
        a = np.abs(np.array(self.ai_voltage_rngs))
        return a[np.argmax(a > max_voltage)]


@dataclass
class TestDoc:
    test_document: str = 'QP10330 rev.0'
    test_software: str = 'CS10XXX rev. X'
    test_software_build: str = '0.0.1'
    data_acq_system: str = 'T0037-01'
    daq_adapter: str = 'T0038-01'
    epg_adapter: str = 'T0018-01'
    tester_name: str = 'M. Faltys'
    epg_main_fw: str = 'CS10107 rev. 6	EPG FW MainMCU'
    epg_watchdog_fw: str = 'CS10111 rev. 6	EPG FW Watchdog MCU'
    epg_ui_fw: str = 'CS10112 rev. 2	EPG FW UI'
    epg_cots_fw: str = 'CS10143 rev. 2	COTS BLE SW'
    cpa_sw: str = 'CS10144 rev. 5	CPA Software'
    epg_main_test_fw: str = 'CS10240 rev. 2	MainMCU Test FW'
    epg_watchdog_test_fw: str = 'WatchdogMCU Test FW'


class AcTest:
    def __init__(self, amp_ua: int, pw_us: int, ipi_us: int, rpr: int, req_freq: float,
                 epg_sn, load_resistance, sample_rate_per_channel):
        self.amp_ua = int(amp_ua)
        self.pw_us = int(round(pw_us / 15, 0) * 15)  # Convert to 15uS increment
        self.ipi_us = int(round(ipi_us / 15, 0) * 15)
        self.rpr = 2 ** np.argmax(np.array([1, 2, 4, 8, 16]) >= rpr)  # 5 possible values
        self.rpw_us = self.pw_us * self.rpr
        self.max_freq = int(self.pw_us + self.ipi_us + self.rpw_us + 15)
        self.frequency = req_freq
        self.epg_sn = epg_sn
        self.load_resistance = load_resistance
        self.sample_rate_per_channel = sample_rate_per_channel
        self.measured_samples_in_cycle = 0

    def name(self):
        return f'{self.epg_sn}_AC_{self.amp_ua:.0f}uA_{self.pw_us}_{self.ipi_us}us_{self.rpr}x_' \
               f'{round(self.frequency, 0):.0f}Hz_{self.load_resistance}ohms'

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, req_freq):
        """Side effects dt1 and period"""
        temp_dt1_us = (1E6 / req_freq) - self.pw_us - self.ipi_us - self.rpw_us
        temp_dt1_us = int(round(temp_dt1_us / 15, 0) * 15)
        self.dt1_us = temp_dt1_us if temp_dt1_us >= 15 else 15
        self.period_us = int(self.pw_us + self.ipi_us + self.rpw_us + self.dt1_us)
        self.period_s = 1e-6 * self.period_us
        self._frequency = 1 / self.period_s

    def measure(self, cycles=1):
        test_data = pd.DataFrame(columns=testset.signals)
        samples_cycle = int(self.sample_rate_per_channel * 1e-6 * self.period_us)
        pre_trigger_samples = int(samples_cycle / 10)
        samples_to_capture = samples_cycle * cycles + 2 * pre_trigger_samples
        print(f'Period: {self.period_us}  Samples:{samples_to_capture} SR: {self.sample_rate_per_channel} '
              f'Duration:{samples_to_capture / self.sample_rate_per_channel}')
        # Capture Data
        for signal in testset.signals[:-1]:  # Don't use Vie because it wont trigger
            trigger_level, voltage_range = self.ac_trigger_range_calculator(signal)
            # Capture Data
            print(f'Waiting for trigger at {trigger_level * testset.v_to_uamv_by_signal[signal]:.0f}uA '
                  f'or {1e3 * trigger_level:.0f}mV')
            dv = testset.get_single_channel(signal, voltage_range=voltage_range,
                                            sample_rate=self.sample_rate_per_channel, samples=samples_to_capture,
                                            triggered=True, trigger_level=trigger_level,
                                            pre_trigger_samples=pre_trigger_samples)
            dv = dv - testset.v_offset_by_signal[signal]  # subtract offsets
            d = dv * testset.v_to_uamv_by_signal[signal]
            test_data[signal] = d
            test_data[signal] = test_data[signal].rolling(window=4, min_periods=1).mean()
        dt = np.linspace(0, samples_to_capture / self.sample_rate_per_channel, samples_to_capture)
        test_data['Time'] = dt
        return test_data

    def ac_trigger_range_calculator(self, signal):
        """Calculate the optimal trigger and DAC voltage range expected"""
        factor = 1.25 if self.rpr > 1 else 0.75  # 25% over recovery pulse
        if signal == 'Iwe1':
            trig_ua = (self.amp_ua / self.rpr) * factor
            trig_v = trig_ua * testset.uamv_to_v_by_signal[signal] / 1e6
            max_v = self.amp_ua * abs(testset.uamv_to_v_by_signal[signal]) * factor / 1e6
        elif signal == 'Iwe2':
            trig_ua = -(self.amp_ua / self.rpr) * factor
            trig_v = trig_ua * testset.uamv_to_v_by_signal[signal] / 1e6
            max_v = self.amp_ua * abs(testset.uamv_to_v_by_signal[signal]) * factor / 1e6
        elif 'Vwe1' in signal:
            trig_ua = (self.amp_ua / self.rpr) * factor
            trig_v = trig_ua * self.load_resistance * testset.amp_gain_by_signal[signal] / 1e6
            max_v = self.amp_ua * self.load_resistance * testset.amp_gain_by_signal[signal] * factor / 1e6
        elif 'Vwe2' in signal:
            trig_ua = -(self.amp_ua / self.rpr) * factor
            trig_v = trig_ua * self.load_resistance * testset.amp_gain_by_signal[signal] / 1e6
            max_v = self.amp_ua * self.load_resistance * testset.amp_gain_by_signal[signal] * factor / 1e6
        elif 'Vie' in signal:
            trig_v = 20 / 1000
            max_v = 10
        else:
            trig_v = 20 / 1000
            max_v = 20 / 1000
        voltage_range = testset.get_minimum_voltage_range(max_v)
        return trig_v, voltage_range


class ZhTest:
    def __init__(self, amp_ua: int, epg_sn, load_resistance, sample_rate_per_channel):
        self.amp_ua = int(amp_ua)
        self.offset_ua = -42
        self.frequency = 1 / 12
        self.epg_sn = epg_sn
        self.load_resistance = load_resistance
        self.sample_rate_per_channel = sample_rate_per_channel
        self.measured_samples_in_cycle = 0

    def name(self):
        return f'{self.epg_sn}_ZH_{self.amp_ua:.0f}uA_{self.offset_ua}uA_{self.load_resistance}ohms'

    def measure(self, cycles=1, filter_window=20):
        """Test ZeroHz capability and publishes a html plot and html report
        :param filter_window: to reduce system noise
        :param cycles: when running chaining tests more than one cycle may be required
        :return: test_data, test_results
        """
        sample_rate = self.sample_rate_per_channel
        samples_cycle_est = int(12.12 * sample_rate)  # assume frequency is off by 10%
        samples = int(samples_cycle_est * (cycles + 1))  # 1 cycle added to assure capture of >1 contiguous cycle
        # 1 Get Waveform and filter
        df_meas = testset.get_iv_data(sample_rate, samples)
        df_meas[:] = df_meas[:].rolling(window=filter_window, min_periods=1, win_type='blackmanharris').mean()
        # Measure frequency measurement so reference can be created at same frequency
        period1 = self.measure_period(df_meas, 'Iwe1', self.offset_ua, cycles)
        period2 = self.measure_period(df_meas, 'Iwe2', self.offset_ua, cycles)
        self.measured_samples_in_cycle = int((period1 + period2) / 2)
        # 2 Calculate Reference Waveform
        df_ref = build_zerohz_waveform_from_template(amp_ua=self.amp_ua,
                                                     samples=self.measured_samples_in_cycle,
                                                     load_resistance=self.load_resistance, offset_ua=self.offset_ua)
        # 3 Extract period based upon best correlation to reference (more reliable than edge or level triggering
        df_meas = self.extract_measurement_that_matches_ref(df_ref, df_meas, 'Iwe1')
        # 4 Build test data
        return calc_error_from_meas_and_ref(df_meas, df_ref, 0)

    @staticmethod
    def measure_period(df, signal, trigger_level=0, cycles=2):
        """Measure the Period of specified channel assuming the data does not produce more than 2 zero crossings
        :param df:
        :param signal:
        :param trigger_level:
        :param cycles: used to determine whether too many zero crossings indicating noisy signal
        :return:
        """
        """Measure the frequency where cycles just validates the number of expected zero crossings"""
        b = np.array(df[signal].values.tolist())  # convert to numpy array

        pos_edges = np.flatnonzero((b[:-1] > trigger_level) & (b[1:] < trigger_level)) + 1
        neg_edges = np.flatnonzero((b[:-1] < trigger_level) & (b[1:] > trigger_level)) + 1
        half_periods = np.diff(np.sort(np.concatenate((neg_edges, pos_edges), axis=0)))
        number_of_half_periods = len(half_periods)
        if number_of_half_periods > cycles * 3:
            print(f'Excessive number of 1/2 periods detected: {number_of_half_periods}, '
                  f'signal may be too noisy for threshold may be set incorrectly')
        return half_periods[-2] + half_periods[-1]  # pick last two as the first could have signal disturbance

    @staticmethod
    def extract_measurement_that_matches_ref(df_ref, df_meas, column):
        """Find the optimal alignment between to frames to minimize the rms error in col
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


def rms_error_between_ref_and_shifted_measurement(df_ref, df_meas, column, shift):
    """ Used recursively in ZhTest Class and could not figure out how to embed in class
    Given a reference shift measurement and measure error for a particular signal of data
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


# Initialize  test document and test set
DEVICE_NAME = 'Dev1'
testdoc = TestDoc()
testset = TestSet(DEVICE_NAME)  # MAKE INSTANCE OF TestSet Class


class TestResultAC:
    """From an expected amplitude, and frequency and a measurement/calculated frame create a test report"""
    def __init__(self, test, test_data):
        from datetime import datetime
        pw_margin = 30
        self.id = test.name()
        self.datetime = datetime.now()
        self.test_software = TestDoc.test_software
        self.test_software_build = TestDoc.test_software_build
        self.cpa_sw = TestDoc.cpa_sw
        self.cpa_sw_build = TestDoc.test_software_build
        self.epg_sn = "FIXME"
        self.daq_adapter = TestDoc.daq_adapter
        self.nidaq_product_type = testset.product_type
        self.nidaq_serial_number = testset.serial_number
        self.df_pp = self.calculate_ac_pulse_parameters(test, test_data)
        self.period = ['1', self.df_pp.loc['Period', 'Iwe1-Meas'], self.df_pp.loc['Period', 'Iwe2-Meas'],
                       (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us) - pw_margin,
                       (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us) + pw_margin,
                       'TECH10074.HW.A1,TECH10074.HW.B1,TECH10074.FW.A1,TECH10074.FW.A13', 'Period', 'uS']
        self.period_error = [self.df_pp.loc['Period', 'Iwe1-Meas'],
                             (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us),
                             self.df_pp.loc['Period', 'Iwe1-Meas']
                             - (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us)]
        self.pw_h = ['2', self.df_pp.loc['PW-H', 'Iwe1-Meas'], self.df_pp.loc['PW-H', 'Iwe2-Meas'],
                     test.pw_us - pw_margin, test.pw_us + pw_margin, 'TECH10074.HW.A3,TECH10074.FW.A10', 'PW-H', 'uS']
        self.pw_l = ['3', self.df_pp.loc['PW-L', 'Iwe1-Meas'], self.df_pp.loc['PW-L', 'Iwe2-Meas'],
                     test.pw_us - pw_margin, test.pw_us + pw_margin, 'TECH10074.FW.A10', 'PW-L', 'uS']
        self.pw_amp = ['4', self.df_pp.loc['PW-AMP', 'Iwe1-Meas'], self.df_pp.loc['PW-AMP', 'Iwe2-Meas'],
                       test.amp_ua * 0.95, test.amp_ua * 1.1,
                       'TECH10074.FW.A11', 'PW-AMP', 'uA']
        self.v_amp = ['13', self.df_pp.loc['V-AMP', 'Iwe1-Meas'], self.df_pp.loc['V-AMP', 'Iwe2-Meas'],
                      1e-3 * test.amp_ua * test.load_resistance * 0.95,
                      1e-3 * test.amp_ua * test.load_resistance * 1.05,
                      'TECH10074.HW.A6', 'V-AMP', 'mV']
        self.ipi = ['5', self.df_pp.loc['IPI', 'Iwe1-Meas'], self.df_pp.loc['IPI', 'Iwe2-Meas'],
                    test.ipi_us - pw_margin, test.ipi_us + pw_margin, 'TECH10074.FW.A11', 'IPI', 'uS']
        self.rt = ['6', self.df_pp.loc['PW-RT', 'Iwe1-Meas'], self.df_pp.loc['PW-RT', 'Iwe2-Meas'],
                   1, 10, 'TECH10074.HW.A5', 'PW-RT', 'uS']
        self.ft = ['7', self.df_pp.loc['PW-FT', 'Iwe1-Meas'], self.df_pp.loc['PW-FT', 'Iwe2-Meas'],
                   1, 10, 'TECH10074.HW.A5', 'PW-FT', 'uS']
        self.rpw_h = ['8', self.df_pp.loc['RPW-H', 'Iwe1-Meas'], self.df_pp.loc['RPW-H', 'Iwe2-Meas'],
                      test.pw_us * test.rpr - pw_margin, test.pw_us * test.rpr + pw_margin, 'TECH10074.FW.A12', 'RPW-H',
                      'uS']
        self.rpw_l = ['9', self.df_pp.loc['RPW-L', 'Iwe1-Meas'], self.df_pp.loc['RPW-L', 'Iwe2-Meas'],
                      test.pw_us * test.rpr - pw_margin, test.pw_us * test.rpr + pw_margin, 'TECH10074.FW.A12', 'RPW-L',
                      'uS']
        self.rpw_amp = ['10', self.df_pp.loc['RPW-AMP', 'Iwe1-Meas'], self.df_pp.loc['RPW-AMP', 'Iwe2-Meas'],
                        (test.amp_ua / test.rpr) * 0.95, (test.amp_ua / test.rpr) * 1.05, 'TECH10074.FW.A12', 'RPW-AMP',
                        'uA']
        self.dc_we = ['11', self.df_pp.loc['DC-WE', 'Iwe1-Meas'], self.df_pp.loc['DC-WE', 'Iwe2-Meas'],
                      -1, 0.5, 'TECH10074.HW.A11', 'DC-WE', 'uA']
        self.dc_ie = ['12', self.df_pp.loc['DC-IE', 'Iwe1-Meas'], self.df_pp.loc['DC-IE', 'Iwe2-Meas'],
                      -1, 0.5, 'TECH10074.HW.A11', 'DC-IE', 'uA']

        self.results = self.create_dataframe()

    def create_dataframe(self):
        """Create a table of the results into a pandas frame so that it can be output in html"""
        return pd.DataFrame([self.__create_dict(self.period),
                             self.__create_dict(self.pw_h),
                             self.__create_dict(self.pw_l),
                             self.__create_dict(self.pw_amp),
                             self.__create_dict(self.v_amp),
                             self.__create_dict(self.ipi),
                             self.__create_dict(self.rt),
                             self.__create_dict(self.ft),
                             self.__create_dict(self.rpw_h),
                             self.__create_dict(self.rpw_l),
                             self.__create_dict(self.rpw_amp),
                             self.__create_dict(self.dc_we),
                             self.__create_dict(self.dc_ie)
                             ])

    def calculate_ac_pulse_parameters(self, test, test_data):
        """Calculate Pulse Parameters for 2 working electrodes in bipolar channel
        :return: df_results where index is parameters and ['Iwe1-Meas', 'Iwe2-Meas'] contains results
        """
        data_dict_1 = self.calculate_ac_pulse_parameters_for_channel(test_data['Iwe1'], test_data['Vwe1_ie'],
                                                                     test_data['Iie'], test,
                                                                     polarity='anodic')
        data_dict_2 = self.calculate_ac_pulse_parameters_for_channel(test_data['Iwe2'], test_data['Vwe2_ie'],
                                                                     test_data['Iie'], test,
                                                                     polarity='cathodic')
        pulse_params = pd.DataFrame(columns=['Iwe1-Meas', 'Iwe2-Meas'])
        pulse_params['Iwe1-Meas'] = pd.Series(data_dict_1)
        pulse_params['Iwe2-Meas'] = pd.Series(data_dict_2)
        return pulse_params

    @staticmethod
    def calculate_ac_pulse_parameters_for_channel(iwe, vwe, iie, test, polarity='anodic'):
        """ Extract the pulse params from AC Spec - tedious
        :param test: class that describes test
        :param vwe:
        :param iie:
        :param iwe:
        :param polarity: 'Anodic' indicates PW is anodic and RPW is cathodic, whereas 'Cathodic' is opposite
        :return: data in data dictionary format
        """
        high = 0.8  # 80%
        low = 0.2  # 20%

        iie = np.array(iie)
        amp_sign = 1  # Pulse timings are setup for anodic pw, just flip the signs on cathodic interface
        iwe = np.array(iwe)
        vwe = np.array(vwe)
        if polarity == 'cathodic':
            iwe = -np.array(iwe)
            vwe = -np.array(vwe)
            amp_sign = -1

        # Use data dictionary to build capture points at 80 and 20% on PW and RPW
        crosses = {'PW-H-POS': 1, 'PW-H-NEG': 1, 'PW-L-POS': 1, 'PW-L-NEG': 1,
                   'RPW-H-POS': 1, 'RPW-H-NEG': 1, 'RPW-L-POS': 1, 'RPW-L-NEG': 1}
        slices = {'PW-H-POS': high * test.amp_ua, 'PW-H-NEG': high * test.amp_ua,
                  'PW-L-POS': low * test.amp_ua, 'PW-L-NEG': low * test.amp_ua,
                  'RPW-H-POS': -high * (test.amp_ua / test.rpr), 'RPW-H-NEG': -high * (test.amp_ua / test.rpr),
                  'RPW-L-POS': -low * (test.amp_ua / test.rpr), 'RPW-L-NEG': -low * (test.amp_ua / test.rpr)}

        for key in slices.keys():  # collect "zero" crossings
            if 'NEG' in key:
                crosses[key] = np.flatnonzero((iwe[:-1] > slices[key]) & (iwe[1:] < slices[key])) + 1
            else:
                crosses[key] = np.flatnonzero((iwe[:-1] < slices[key]) & (iwe[1:] > slices[key])) + 1

        to_us = 1.0e6 / test.sample_rate_per_channel
        _10_uS = int(20 * test.sample_rate_per_channel / 1e6)
        # extract all parameters on each trace
        data = {'Period': 1e6 * (crosses['PW-H-POS'][1] - crosses['PW-H-POS'][0]) / test.sample_rate_per_channel,
                'PW-H': (crosses['PW-H-NEG'][0] - crosses['PW-H-POS'][0]) * to_us,
                'PW-L': (crosses['PW-L-NEG'][0] - crosses['PW-L-POS'][0]) * to_us,
                'PW-AMP': amp_sign * np.mean(iwe[crosses['PW-H-POS'][0]: crosses['PW-H-NEG'][0]]),
                'V-AMP': amp_sign * np.max(vwe[crosses['PW-H-POS'][0]: crosses['PW-H-NEG'][0]]),
                'IPI': (crosses['RPW-L-NEG'][0] - crosses['PW-L-NEG'][0]) * to_us,
                'PW-RT': (crosses['PW-H-POS'][0] - crosses['PW-L-POS'][0]) * to_us,
                'PW-FT': (crosses['PW-L-NEG'][0] - crosses['PW-H-NEG'][0]) * to_us,
                'RPW-H': (crosses['RPW-H-POS'][0] - crosses['RPW-H-NEG'][0]) * to_us,
                'RPW-AMP': amp_sign * np.mean(iwe[crosses['RPW-H-NEG'][0]: crosses['RPW-H-POS'][0]]),
                'RPW-L': (crosses['RPW-L-POS'][0] - crosses['RPW-L-NEG'][0]) * to_us,
                'RPW-RT': (crosses['RPW-L-POS'][0] - crosses['RPW-H-POS'][0]) * to_us,
                'RPW-FT': (crosses['RPW-H-NEG'][0] - crosses['RPW-L-NEG'][0]) * to_us,
                'DT1': (crosses['PW-L-POS'][1] - crosses['RPW-L-POS'][0]) * to_us,
                'DT1-AMP': amp_sign * np.mean(iwe[crosses['RPW-L-POS'][0]:crosses['PW-L-POS'][1]]),
                'DC-WE': amp_sign * np.mean(iwe[crosses['PW-L-POS'][0] - _10_uS:crosses['PW-L-POS'][1] - _10_uS]),
                'DC-IE': amp_sign * np.mean(iie[crosses['PW-L-POS'][0] - _10_uS:crosses['PW-L-POS'][1] - _10_uS])
                }
        return data

    @staticmethod
    def __create_dict(rl):
        """Given measurement and expected range create a row in frame
        :param rl: [Test ID, Measured 1, Measured 2, Range A, Range B, Requirement Trace, Signal, Unit]
        """
        meas1 = (rl[3] <= abs(rl[1]) <= rl[4]) | (rl[3] >= abs(rl[1]) >= rl[4])
        meas2 = (rl[3] <= abs(rl[2]) <= rl[4]) | (rl[3] >= abs(rl[2]) >= rl[4])
        result = 'Pass' if (meas1 & meas2) else 'Fail'  # in between
        return {'TestID': rl[0], 'Parameter': rl[6], 'Units': rl[7],
                'Meas(We1)': round(rl[1], 1), 'Meas(We2)': round(rl[2], 1),
                'RangeA': round(rl[3], 1), 'RangeB': round(rl[4], 1),
                'Pass/Fail': result, 'Requirement': rl[5]}

    def print_report(self):
        """Print HTML report"""
        f = open(data_path() + self.id + '_report.html', 'w')
        lines = [
            f"<!DOCTYPE html>",
            f"<html><head><h2>Test Report</h2></head>",
            f"<body><p>Test ID: {self.id}</p>",
            f"<body><p>Files - Results :{self.id}_report.html,   Data File:_.csv,   Plot File:_.html</p>",
            f"<p>Reference Document: {TestDoc.test_document}</p>",
            f"<p>Date/Time: {self.datetime}</p>",
            f"<p>EPG SN: FIXME </p>",
            f"<p>CPA: {TestDoc.cpa_sw}</p>",
            f"<p>DAQ Model #: {self.nidaq_product_type},  DAQ Tool #: {TestDoc.data_acq_system},"
            f"  Adapter Tool #: {TestDoc.daq_adapter}</p>",
            f"<p>Test Set Software: {TestDoc.test_software},  "
            f"Build ID: {TestDoc.test_software_build}</p>",
            f"</body></html>"
        ]
        for line in lines:
            f.write(line)
        message = self.results.to_html()
        f.write(message)
        f.close()


class TestResultZH:
    """From an expected amplitude, and frequency and a measurement/calculated frame create a test report"""

    # def __init__(self, test_title, amplitude_ua, df, frequency):
    def __init__(self, test, df):
        from datetime import datetime
        self.id = test.name()
        self.datetime = datetime.now()
        self.test_software = TestDoc.test_software
        self.test_software_build = TestDoc.test_software_build
        self.cpa_sw = TestDoc.cpa_sw
        self.cpa_sw_build = TestDoc.test_software_build
        self.epg_sn = "FIXME"
        self.daq_adapter = TestDoc.daq_adapter
        self.nidaq_product_type = testset.product_type
        self.nidaq_serial_number = testset.serial_number
        self.peak_amp_ua = test.amp_ua
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
            ['14', (test.sample_rate_per_channel / test.measured_samples_in_cycle) * 1000, 1000 / (12 * 0.99),
             1000 / (12 * 1.01), 'TECH10074.A9', 'Frequency', 'mHz']
        self.results = self.create_dataframe()

    def create_dataframe(self):
        """Create a table of the results into a pandas frame so that it can be output in html"""
        return pd.DataFrame([self.__create_dict(self.iwe1_max),
                             self.__create_dict(self.iwe1_min),
                             self.__create_dict(self.rms_err_Iwe1),
                             self.__create_dict(self.iwe2_max),
                             self.__create_dict(self.iwe2_min),
                             self.__create_dict(self.rms_err_Iwe2),
                             self.__create_dict(self.iie_max),
                             self.__create_dict(self.iie_min),
                             self.__create_dict(self.rms_err_Iie),
                             self.__create_dict(self.vwe1_max),
                             self.__create_dict(self.vwe1_min),
                             self.__create_dict(self.vwe1_max),
                             self.__create_dict(self.vwe2_min),
                             self.__create_dict(self.frequency)])

    @staticmethod
    def __create_dict(rl):
        """Given measurement and expected range create a row in frame
        :param rl: [Test ID, Measured, Range A, Range B, Requirement Trace, Signal, Unit]
        """
        result = 'Pass' if (rl[2] <= rl[1] <= rl[3]) | (rl[2] >= rl[1] >= rl[3]) else 'Fail'  # in between
        results = pd.Series({'TestID': rl[0], 'Parameter': rl[5], 'Units': rl[6],
                             'Measured': round(rl[1], 1), 'RangeA': round(rl[2], 1), 'RangeB': round(rl[3], 1),
                             'Pass/Fail': result, 'Requirement': rl[4]}, )
        return results

    def print_report(self):
        """Print HTML report"""
        f = open(data_path() + self.id + '_report.html', 'w')
        lines = [
            f"<!DOCTYPE html>",
            f"<html><head><h2>Test Report</h2></head>",
            f"<body><p>Test ID: {self.id}</p>",
            f"<body><p>Files - Results :{self.id}_report.html,   Data File:_.csv,   Plot File:_.html</p>",
            f"<p>Reference Document: {TestDoc.test_document}</p>",
            f"<p>Date/Time: {self.datetime}</p>",
            f"<p>EPG SN: FIXME </p>",
            f"<p>CPA: {TestDoc.cpa_sw}</p>",
            f"<p>DAQ Model #: {self.nidaq_product_type},  DAQ Tool #: {TestDoc.data_acq_system},"
            f"  Adapter Tool #: {TestDoc.daq_adapter}</p>",
            f"<p>Test Set Software: {TestDoc.test_software},  "
            f"Build ID: {TestDoc.test_software_build}</p>",
            f"</body></html>"
        ]
        for line in lines:
            f.write(line)
        message = self.results.to_html()
        f.write(message)
        f.close()


def measurement_studio_measurement(filename):
    df = pd.read_csv(filename)
    gains = testset.v_to_uamv

    for cidx, col in enumerate(df.columns[1:-1]):  # dont touch Time Column
        for idx in df.index:
            string_rep = str(df.loc[idx, col])
            if 'm' in string_rep:
                df.loc[idx, col] = float(string_rep.rstrip('m')) * 1e-3 * gains[cidx]
            elif 'u' in string_rep:
                df.loc[idx, col] = float(string_rep.rstrip('u')) * 1e-6 * gains[cidx]
            else:
                df.loc[idx, col] = float(string_rep) * gains[cidx]
    df.to_pickle('MeasurementStudioReferenceCapture_5000uA_634ohms')
    return df


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
                      trigger_level=-100):
    """Take waveform, find trigger and trim of in front of trigger
    :param df:
    :param signal: signal in frame either integer or string
    :param sample_rate: used to compute frequency and pulse width
    :param pre_trigger_samples: samples to keep before trigger
    :param post_trigger_samples: samples to keep after trigger
    :param trigger_level:
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
    charge_columns = []  # hold charge signal names
    sample_period = (df.index[-1] - df.index[0]) / len(df)
    list_current_signals = [i for i in testset.signals if 'I' in i]
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

    output_file(f'{data_path()}{plot_title}.html')
    if df.shape[1] > 2:
        colors = Category20[df.shape[1]]  # limited to 20 colors
    else:
        colors = ['#1f77b4', '#aec7e8']  # why do palettes have a min of 3?
    p = figure(title=plot_title, x_axis_label='Time(s)', y_axis_label=left_axis_label,
               x_range=(df.Time[0], df.Time.iloc[-1] * 1.3))
    for i, col in enumerate(df.columns):
        if col != 'Time':
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


def build_zerohz_waveform_from_template(amp_ua=1000, offset_ua=-42, samples=12000, load_resistance=2000,
                                        template_file='12s(2-2-2)-SPRe.csv', template_duration=12,
                                        template_sample_rate=100):
    """Creates ZeroHz waveform from csv template and returns dataframe of samples requested
    :param amp_ua: scale amplitude in uA units
    :param offset_ua: in
    :param samples: samples to generate where sample rate of reference where output
                    sample rate = samples / template_duration
    :param load_resistance: only handles resistive loads
    :param template_file: CSV file format one signal of data only
    :param template_duration: non-biased waveform where abs() is <= 1
    :param template_sample_rate: rate at which CSV is sampled since it has no time
    :return: generated data frame with the following columns ['Time', 'Iwe1', 'Iwe2', 'Iie', 'Vwe1', 'Vwe2']
    """
    from scipy import signal

    dr = pd.read_csv(template_file, header=None)
    template_samples = dr.shape[0]
    # This reference file already has been scaled down to offset assume 1 ma, the the scale factors need tweaked
    scale = (amp_ua - 21) / (1.0 - 0.021)

    # Build reference frame at its native sample rate
    dr['Time'] = np.linspace(0, template_duration - 1 / template_sample_rate, template_samples)
    dr['Iwe1'] = dr[0] * scale + (offset_ua / 2)  # dr[0] raw data
    dr['Iwe2'] = -dr[0] * scale + (offset_ua / 2)
    dr['Iie'] = offset_ua
    dr = dr.drop([0], axis=1)
    dr = dr.reset_index(drop=True)

    # build data at new sample rate
    new_time = np.linspace(0, template_duration - (1 / samples), samples)
    new_iwe1 = signal.resample(dr.Iwe1, samples)
    new_iwe2 = signal.resample(dr.Iwe2, samples)

    new_vwe1 = (new_iwe1 * load_resistance) / 1000.0
    new_vwe2 = (new_iwe2 * load_resistance) / 1000.0
    # build pandas data frame
    df = pd.DataFrame(columns=['Time', 'Iwe1', 'Iwe2', 'Iie', 'Vwe1', 'Vwe2'])
    df['Time'] = new_time
    df['Iwe1'] = new_iwe1
    df['Iwe2'] = new_iwe2
    df['Iie'] = offset_ua
    df['Vwe1'] = new_vwe1
    df['Vwe2'] = new_vwe2
    iwe1_min = df['Iwe1'].min()
    iwe1_max = df['Iwe1'].max()
    iwe2_min = df['Iwe2'].min()
    iwe2_max = df['Iwe2'].max()

    print(f'iwe1_min:{iwe1_min} iwe1_max:{iwe1_max} iwe2_min:{iwe2_min} iwe2_max:{iwe2_max}')

    return df


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
    df = testset.get_iv_data(sample_rate, samples)
    capture_boundary = int(samples_cycle / 200)  # show 1% of waveform before and after waveform
    df[:] = df[:].rolling(window=window, min_periods=1, win_type='blackmanharris').mean()
    dft, frequency, pulsewidth = \
        _software_trigger(df, trigger_channel, sample_rate,
                          trigger_level=-int(trigger_level_percent * maximum_current_ma * 1000 / 100),
                          post_trigger_samples=samples_cycle * cycles + 2 * capture_boundary,
                          pre_trigger_samples=capture_boundary)

    plot_data_frame_left_right(dft, plot_title,
                               'Current (uA)', [i for i in testset.signals if 'I' in i],
                               'Voltage(mV)', [i for i in testset.signals if 'V' in i])
    return df, dft, frequency, pulsewidth


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
