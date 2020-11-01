"""
import pmtest
from pmtest import TestSet
testset = TestSet('Dev1')
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
import time
from datetime import datetime
from dataclasses import dataclass


def data_path():
    # return os.getcwd() + "\\Data\\"   # TODO does not work in debug mode
    return "C:\\Users\\mike\\OneDrive\\Documents\\GitHub\\MultimodeEPGTest\\Data"


class TestSet:
    """ Provides test set functionality and configuration for strategy capture
    """

    def __init__(self, idx):
        # Manual definition of test set hardware
        self.signals = ['Iwe1', 'Iwe2', 'Iie', 'Vwe1_ie', 'Vwe2_ie']
        self.physical_channels = ['ai13', 'ai7', 'ai15', 'ai5', 'ai6']  # T0038
        self.sense_resistors = [10, 10, -49.9, 1000, 1000]
        self.amp_gains = [60.1716, 60.1716, 79.8644, 1, 1]
        self.load_resistances = [100, 500, 1000, 10000]
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
        print(f'{self.product_type} found named: {idx}')
        self.num_channels = len(self.signals)
        self.v_offset = [0, 0, 0, 0, 0]
        self.max_sample_rate_per_channel = int(self.max_single_chan_rate / self.num_channels)
        self.sample_rate_per_channel = self.max_sample_rate_per_channel  # set to max sample rate unless overridden
        self.physical_channel_string = ', '.join([self.name + '/' + i for i in self.physical_channels])
        self.physical_channel_by_signal = dict(zip(self.signals, self.physical_channel_string.split(',')))
        self.v_to_uamv = [1e6 / (gain * res) for gain, res in zip(self.amp_gains, self.sense_resistors)]
        self.uamv_to_v = [(gain * res) for gain, res in zip(self.amp_gains, self.sense_resistors)]
        self.v_offset_by_signal = dict(zip(self.signals, self.v_offset))
        self.v_to_uamv_by_signal = dict(zip(self.signals, self.v_to_uamv))
        self.uamv_to_v_by_signal = dict(zip(self.signals, self.uamv_to_v))
        self.amp_gain_by_signal = dict(zip(self.signals, self.amp_gains))
        self.sense_resistor_by_signal = dict(zip(self.signals, self.sense_resistors))
        self._load = 'LdExt'
        self._channel_pair = "E1_2"
        self._load_hiZ = 'OFF'
        self._ie_connect = 'ON'
        self._relay_led = 'OFF'
        print(f'Signals: {self.signals} being measured')

    @property
    def load_hiZ(self):
        return self._load_hiZ

    @load_hiZ.setter
    def load_hiZ(self, value):
        if value == "ON":
            self.set_relay('LdExtHiZ', 'ON')
        else:
            self.set_relay('LdExtHiZ', 'OFF')

    @property
    def ie_connect(self):
        return self._ie_connect

    @ie_connect.setter
    def ie_connect(self, value):
        if value == "ON":
            self.set_relay('DisIE', 'OFF')
        else:
            self.set_relay('DisIE', 'ON')

    @property
    def relay_led(self):
        return self._relay_led

    @relay_led.setter
    def relay_led(self, value):
        if value == "ON":
            self.set_digital_port0(int('10000000', 2))
        else:
            self.set_digital_port0(int('00000000', 2))

    @property
    def channel_pair(self):
        return self._channel_pair

    @channel_pair.setter
    def channel_pair(self, pair):
        """Set Channel Pair where there are 8 pairs and pair 1-2 are on by default
         :param self:
         :param pair: 0: 'All_Off', 1-2,... 15-16, >16 'All_On'
         """
        'E15_16', 'E13_14', 'E11_12', 'E9_10', 'E7_8', 'E5_6', 'E3_4', 'E1_2'
        if pair == 'All_Off':
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E1_2':
            self.set_relay('E1_2', 'ON')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E3_4':
            self.set_relay('E3_4', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E5_6':
            self.set_relay('E5_6', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E7_8':
            self.set_relay('E7_8', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E9_10':
            self.set_relay('E9_10', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E11_12':
            self.set_relay('E11_12', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E13_14', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E13_14':
            self.set_relay('E13_14', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E15_16', 'OFF')
        elif pair == 'E15_16':
            self.set_relay('E15_16', 'ON')
            self.set_relay('E1_2', 'OFF')
            self.set_relay('E3_4', 'OFF')
            self.set_relay('E5_6', 'OFF')
            self.set_relay('E7_8', 'OFF')
            self.set_relay('E9_10', 'OFF')
            self.set_relay('E11_12', 'OFF')
            self.set_relay('E13_14', 'OFF')
        elif pair == 'All_On':
            self.set_relay('E1_2', 'ON')
            self.set_relay('E3_4', 'ON')
            self.set_relay('E5_6', 'ON')
            self.set_relay('E7_8', 'ON')
            self.set_relay('E9_10', 'ON')
            self.set_relay('E11_12', 'ON')
            self.set_relay('E13_14', 'ON')
            self.set_relay('E15_16', 'ON')
        else:
            print("Illegal Channel Name")

    @property
    def load(self):
        return self._load

    @load.setter
    def load(self, load_value):
        """Set Load Resistance
         :param self:
         :param load_value: OPEN, 100, 500, 1000, 2000
         Add load before removing previous
         """
        if load_value == 100:
            self.set_relay('Ld100', 'ON')
            self.set_relay('Ld500', 'OFF')
            self.set_relay('Ld1K', 'OFF')
            self.set_relay('Ld2K', 'OFF')
            self.set_relay('LdExt', 'OFF')
        elif load_value == 500:
            self.set_relay('Ld500', 'ON')
            self.set_relay('Ld100', 'OFF')
            self.set_relay('Ld1K', 'OFF')
            self.set_relay('Ld2K', 'OFF')
            self.set_relay('LdExt', 'OFF')
        elif load_value == 1000:
            self.set_relay('Ld1K', 'ON')
            self.set_relay('Ld100', 'OFF')
            self.set_relay('Ld500', 'OFF')
            self.set_relay('Ld2K', 'OFF')
            self.set_relay('LdExt', 'OFF')
        elif load_value == 2000:
            self.set_relay('Ld2K', 'ON')
            self.set_relay('Ld100', 'OFF')
            self.set_relay('Ld500', 'OFF')
            self.set_relay('Ld1K', 'OFF')
            self.set_relay('LdExt', 'OFF')
        elif load_value == 'Open':
            self.set_relay('Ld100', 'OFF')
            self.set_relay('Ld500', 'OFF')
            self.set_relay('Ld1K', 'OFF')
            self.set_relay('Ld2K', 'OFF')
            self.set_relay('LdExt', 'OFF')
        elif load_value == 'LdExt':
            self.set_relay('LdExt', 'ON')
            self.set_relay('Ld100', 'OFF')
            self.set_relay('Ld500', 'OFF')
            self.set_relay('Ld1K', 'OFF')
            self.set_relay('Ld2K', 'OFF')
        else:
            print("Illegal Load Value")
        self._load = load_value

    def set_digital_port0(self, lines):
        """
        Sets all 8 Bits of Digital Port simultaneously
        :param lines:
        :return:
        """
        digital_output = f'{self.name}/port0/line0:7'
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(digital_output, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
            try:
                task.write(lines, auto_start=True)
                # task.stop()
            except nidaqmx.DaqError as e:
                print(e)

    def set_relay(self, relay, state):
        """
        Sets relays (1-16), relay 0 turns them all off through MAX4821 Part, relay 16 is a LED
        Bit 0: CS1n (Relays 9-16)
        Bit 1: RESETn resets all relays
        Bit 2: CS0n (Relays 1-8)
        Bit 3: LVL - High turns on relay, Low Turns off relay
        Bit 4-7: Address of relay
        :param state:
        :param relay:
        :return:
        """
        relays = ['E15_16', 'E13_14', 'E11_12', 'E9_10', 'E7_8', 'E5_6', 'E3_4', 'E1_2',
                  'LdExtHiZ', 'LdExt', 'P10', 'DisIE', 'Ld2K', 'Ld1K', 'Ld500', 'Ld100', 'Default']
        relay_idx = relays.index(relay)

        RESETn = 1 if relay_idx != 17 else 0
        CS1n = 0 if relay_idx > 7 else 1
        CS0n = 0 if relay_idx < 8 else 1

        invert_state = True if (relay == 'E1_2') or (relay == 'LdExt') else False  # These relays wired on
        LVL = 1 if state == 'ON' else 0
        LVL = LVL ^ invert_state
        ADR = relay_idx if relay_idx < 8 else relay_idx - 8
        lines = CS1n + RESETn * 2 + CS0n * 4 + LVL * 8 + ADR * 16
        self.set_digital_port0(lines)
        print(f'{relay} : {LVL} : {lines:>08b}')
        CS1n = 1
        CS0n = 1
        RESETn = 1
        # LVL = LVL ^ 1
        lines = CS1n + RESETn * 2 + CS0n * 4  # + LVL * 8 + ADR * 16
        self.set_digital_port0(lines)
        print(f'{lines:>08b}')

    def calibrate_daq_adapter_offset(self):
        """Offset Measured when there are no inputs present
        and load is at 1Kohm
        :return:
        """
        self.load = 1000
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
        self.v_offset = np.mean(da, 0)
        self.v_offset_by_signal = dict(zip(self.signals, self.v_offset))
        print(f'Raw Offset Values: {self.v_offset * 1e6} uV')
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

    def get_all_channels(self, sample_rate, samples, voltage_range=10):
        """Get current, voltage data from first Nidaq board in list
        :param sample_rate:
        :param samples:
        :return: df frame
        """
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
        # transform voltages to currents
        dt = np.linspace(0, samples / sample_rate, samples)
        da = da - self.v_offset
        print(da.shape, self.v_to_uamv)
        da = da * np.array(self.v_to_uamv)  # C
        columns = ['Time'] + self.signals
        da = np.column_stack((dt, da))
        df = pd.DataFrame(data=da[0:, 0:], columns=columns)
        return df

    def get_single_channel(self, signal, voltage_range=10.0, sample_rate=1e6, samples=50000,
                           triggered=False, trigger_level=1, pre_trigger_samples=0):
        """ Get single channel measurement without any conditioning in volts
        :param signal:
        :param voltage_range:
        :param sample_rate:
        :param samples:
        :param triggered:
        :param trigger_level:
        :param pre_trigger_samples:
        :return:
        """
        physical_channel = self.physical_channel_by_signal[signal]
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
    def __init__(self, testset, amp_ua: int, pw_us: int, ipi_us: int, rpr: int, req_freq: float,
                 epg_sn, load, sample_rate_per_channel):
        self.testset = testset
        self.amp_ua = int(amp_ua)
        self.pw_us = int(round(pw_us / 15, 0) * 15)  # Convert to 15uS increment
        self.ipi_us = int(round(ipi_us / 15, 0) * 15)
        self.rpr = 2 ** np.argmax(np.array([1, 2, 4, 8, 16]) >= rpr)  # 5 possible values
        self.rpw_us = self.pw_us * self.rpr
        self.max_freq = int(self.pw_us + self.ipi_us + self.rpw_us + 15)
        self.frequency = req_freq
        self.epg_sn = epg_sn
        self.load = load
        self.sample_rate_per_channel = sample_rate_per_channel
        self.measured_samples_in_cycle = 0
        self.data = None

    def name(self):
        return f'{self.epg_sn}_AC_{self.amp_ua:.0f}uA_{self.pw_us}_{self.ipi_us}us_{self.rpr}x_' \
               f'{round(self.frequency, 0):.0f}Hz_{self.load}ohms'

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
        self.data = pd.DataFrame(columns=self.testset.signals)
        samples_cycle = int(self.sample_rate_per_channel * 1e-6 * self.period_us)
        pre_trigger_samples = int(samples_cycle / 10)
        samples_to_capture = samples_cycle * cycles + 2 * pre_trigger_samples
        print(f'Period: {self.period_us}  Samples:{samples_to_capture} SR: {self.sample_rate_per_channel} '
              f'Duration:{samples_to_capture / self.sample_rate_per_channel}')
        # Capture Data
        for signal in self.testset.signals:
            trigger_level, voltage_range = self.ac_trigger_range_calculator(signal)
            # Capture Data
            print(f'Waiting for trigger at {trigger_level * self.testset.v_to_uamv_by_signal[signal]:.0f}uA '
                  f'or {1e3 * trigger_level:.0f}mV')
            dv = self.testset.get_single_channel(signal, voltage_range=voltage_range,
                                                 sample_rate=self.sample_rate_per_channel, samples=samples_to_capture,
                                                 triggered=True, trigger_level=trigger_level,
                                                 pre_trigger_samples=pre_trigger_samples)
            dv = dv - self.testset.v_offset_by_signal[signal]  # subtract offsets
            d = dv * self.testset.v_to_uamv_by_signal[signal]
            self.data[signal] = d
            self.data[signal] = self.data[signal].rolling(window=4, min_periods=1).mean()
        dt = np.linspace(0, samples_to_capture / self.sample_rate_per_channel, samples_to_capture)
        self.data['Time'] = dt

    def ac_trigger_range_calculator(self, signal):
        """Calculate the optimal trigger and DAC voltage range expected"""
        factor = 1.25
        trig_factor = factor if self.rpr > 1 else 0.75  # 25% over recovery pulse
        if signal == 'Iwe1':
            trig_ua = (self.amp_ua / self.rpr) * trig_factor
            trig_v = trig_ua * self.testset.uamv_to_v_by_signal[signal] / 1e6
            max_v = self.amp_ua * abs(self.testset.uamv_to_v_by_signal[signal]) * factor / 1e6
        elif signal == 'Iwe2':
            trig_ua = -(self.amp_ua / self.rpr) * trig_factor
            trig_v = trig_ua * self.testset.uamv_to_v_by_signal[signal] / 1e6
            max_v = self.amp_ua * abs(self.testset.uamv_to_v_by_signal[signal]) * factor / 1e6
        elif 'Vwe1' in signal:
            trig_ua = (self.amp_ua / self.rpr) * trig_factor
            trig_v = trig_ua * self.load * self.testset.amp_gain_by_signal[signal] / 1e6
            max_v = self.amp_ua * self.load * self.testset.amp_gain_by_signal[signal] * factor / 1e6
        elif 'Vwe2' in signal:
            trig_ua = -(self.amp_ua / self.rpr) * trig_factor
            trig_v = trig_ua * self.load * self.testset.amp_gain_by_signal[signal] / 1e6
            max_v = self.amp_ua * self.load * self.testset.amp_gain_by_signal[signal] * factor / 1e6
        elif 'Vie' in signal:
            trig_v = 20 / 1000
            max_v = 10
        else:
            trig_v = 20 / 1000
            max_v = 20 / 1000
        voltage_range = self.testset.get_minimum_voltage_range(max_v)
        return trig_v, voltage_range


class ZhTest:
    def __init__(self, testset, amp_ua: int, epg_sn, load, sample_rate_per_channel):
        self.testset = testset
        self.amp_ua = int(amp_ua)
        self.offset_ua = -42
        self.frequency = 1 / 12
        self.epg_sn = epg_sn
        self.load = load
        self.sample_rate_per_channel = sample_rate_per_channel
        self.measured_samples_in_cycle = 0
        self.data = None

    def name(self):
        return f'{self.epg_sn}_ZH_{self.amp_ua:.0f}uA_{self.offset_ua}uA_{self.load}ohms'

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
        df_meas = self.testset.get_all_channels(sample_rate, samples)
        df_meas[:] = df_meas[:].rolling(window=filter_window, min_periods=1, win_type='blackmanharris').mean()
        # Measure frequency measurement so reference can be created at same frequency
        period1 = self.measure_period(df_meas, 'Iwe1', self.offset_ua, cycles)
        period2 = self.measure_period(df_meas, 'Iwe2', self.offset_ua, cycles)
        self.measured_samples_in_cycle = int((period1 + period2) / 2)
        # 2 Calculate Reference Waveform
        df_ref = build_zerohz_waveform_from_template(amp_ua=self.amp_ua,
                                                     samples=self.measured_samples_in_cycle,
                                                     load=self.load, offset_ua=self.offset_ua)
        # 3 Extract period based upon best correlation to reference (more reliable than edge or level triggering
        df_meas = self.extract_measurement_that_matches_ref(df_ref, df_meas, 'Iwe1')
        # 4 Build test data
        self.data = self.calc_error_from_meas_and_ref(df_meas, df_ref, 0)

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

    @staticmethod
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




def build_zerohz_waveform_from_template(amp_ua=1000, offset_ua=-42, samples=12000, load=2000,
                                        template_file='12s(2-2-2)-SPRe.csv', template_duration=12,
                                        template_sample_rate=100):
    """Creates ZeroHz waveform from csv template and returns dataframe of samples requested
    :param amp_ua: scale amplitude in uA units
    :param offset_ua: in
    :param samples: samples to generate where sample rate of reference where output
                    sample rate = samples / template_duration
    :param load: only handles resistive loads
    :param template_file: CSV file format one signal of data only
    :param template_duration: non-biased waveform where abs() is <= 1
    :param template_sample_rate: rate at which CSV is sampled since it has no time
    :return: generated data frame with the following columns ['Time', 'Iwe1', 'Iwe2', 'Iie', 'Vwe1', 'Vwe2']
    """
    from scipy import signal

    dr = pd.read_csv(data_path() + "\\" + template_file, header=None)
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

    new_vwe1 = (new_iwe1 * load) / 1000.0
    new_vwe2 = (new_iwe2 * load) / 1000.0
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


class TestResultAC:
    """From an expected amplitude, and frequency and a measurement/calculated frame create a test report"""

    def __init__(self, test):
        from datetime import datetime
        terr_l = 0.985  # Timing error based on 1.5% accurate clock
        terr_h = 1.015
        toff = 15  # Timing offset based on 15 uS timing resolution
        aerr_l = 0.95  # Amplitude error set to 5% for AC
        aerr_h = 1.05
        aoff = 25  # Amplitude offset of 25uA
        self.id = test.name()
        self.datetime = datetime.now()
        self.test_document = TestDoc.test_document
        self.test_software = TestDoc.test_software
        self.test_software_build = TestDoc.test_software_build
        self.cpa_sw = TestDoc.cpa_sw
        self.cpa_sw_build = TestDoc.test_software_build
        self.nidaq_product_type = test.testset.product_type
        self.data_acq_system = TestDoc.data_acq_system
        self.daq_adapter = TestDoc.daq_adapter
        self.df_pp = self.calculate_ac_pulse_parameters(test)
        self.period = ['1', self.df_pp.loc['Period', 'Iwe1-Meas'], self.df_pp.loc['Period', 'Iwe2-Meas'],
                       (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us) * terr_l - toff,
                       (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us) * terr_h + toff,
                       'TECH10074.HW.A1,TECH10074.HW.B1,TECH10074.FW.A1,TECH10074.FW.A13', 'Period', 'uS']
        self.period_error = [self.df_pp.loc['Period', 'Iwe1-Meas'],
                             (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us),
                             self.df_pp.loc['Period', 'Iwe1-Meas']
                             - (test.pw_us * (1 + test.rpr) + test.ipi_us + test.dt1_us)]
        self.pw_h = ['2', self.df_pp.loc['PW-H', 'Iwe1-Meas'], self.df_pp.loc['PW-H', 'Iwe2-Meas'],
                     test.pw_us * terr_l - toff, test.pw_us * terr_h + toff,
                     'TECH10074.HW.A3,TECH10074.FW.A10', 'PW-H', 'uS']
        self.pw_l = ['3', self.df_pp.loc['PW-L', 'Iwe1-Meas'], self.df_pp.loc['PW-L', 'Iwe2-Meas'],
                     test.pw_us * terr_l - toff, test.pw_us * terr_h + toff, 'TECH10074.FW.A10', 'PW-L', 'uS']
        self.pw_amp = ['4', self.df_pp.loc['PW-AMP', 'Iwe1-Meas'], self.df_pp.loc['PW-AMP', 'Iwe2-Meas'],
                       test.amp_ua * aerr_l - aoff, test.amp_ua * aerr_h + aoff,
                       'TECH10074.FW.A11', 'PW-AMP', 'uA']
        self.v_amp = ['13', self.df_pp.loc['V-AMP', 'Iwe1-Meas'], self.df_pp.loc['V-AMP', 'Iwe2-Meas'],
                      1e-3 * test.amp_ua * test.load * aerr_l - aoff,
                      1e-3 * test.amp_ua * test.load * aerr_h + aoff,
                      'TECH10074.HW.A6', 'V-AMP', 'mV']
        self.ipi = ['5', self.df_pp.loc['IPI', 'Iwe1-Meas'], self.df_pp.loc['IPI', 'Iwe2-Meas'],
                    test.ipi_us * terr_l - toff, test.ipi_us * terr_h + toff, 'TECH10074.FW.A11', 'IPI', 'uS']
        self.rt = ['6', self.df_pp.loc['PW-RT', 'Iwe1-Meas'], self.df_pp.loc['PW-RT', 'Iwe2-Meas'],
                   1, 10, 'TECH10074.HW.A5', 'PW-RT', 'uS']
        self.ft = ['7', self.df_pp.loc['PW-FT', 'Iwe1-Meas'], self.df_pp.loc['PW-FT', 'Iwe2-Meas'],
                   1, 10, 'TECH10074.HW.A5', 'PW-FT', 'uS']
        self.rpw_h = ['8', self.df_pp.loc['RPW-H', 'Iwe1-Meas'], self.df_pp.loc['RPW-H', 'Iwe2-Meas'],
                      test.pw_us * test.rpr * terr_l - toff, test.pw_us * test.rpr * terr_h + toff, 'TECH10074.FW.A12',
                      'RPW-H', 'uS']
        self.rpw_l = ['9', self.df_pp.loc['RPW-L', 'Iwe1-Meas'], self.df_pp.loc['RPW-L', 'Iwe2-Meas'],
                      test.pw_us * test.rpr * terr_l - toff, test.pw_us * test.rpr * terr_h + toff, 'TECH10074.FW.A12',
                      'RPW-L', 'uS']
        self.rpw_amp = ['10', self.df_pp.loc['RPW-AMP', 'Iwe1-Meas'], self.df_pp.loc['RPW-AMP', 'Iwe2-Meas'],
                        (test.amp_ua / test.rpr) * aerr_l - toff, (test.amp_ua / test.rpr) * aerr_h + toff,
                        'TECH10074.FW.A12', 'RPW-AMP', 'uA']
        self.dc_we = ['11', self.df_pp.loc['DC-WE', 'Iwe1-Meas'], self.df_pp.loc['DC-WE', 'Iwe2-Meas'],
                      -1, 1, 'TECH10074.HW.A11', 'DC-WE', 'uA']
        self.dc_ie = ['12', self.df_pp.loc['DC-IE', 'Iwe1-Meas'], self.df_pp.loc['DC-IE', 'Iwe2-Meas'],
                      -1, 1, 'TECH10074.HW.A11', 'DC-IE', 'uA']

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

    def calculate_ac_pulse_parameters(self, test):
        """Calculate Pulse Parameters for 2 working electrodes in bipolar channel
        :return: df_results where index is parameters and ['Iwe1-Meas', 'Iwe2-Meas'] contains results
        """
        data_dict_1 = self.calculate_ac_pulse_parameters_for_channel(test.data['Iwe1'], test.data['Vwe1_ie'],
                                                                     test.data['Iie'], test,
                                                                     polarity='anodic')
        data_dict_2 = self.calculate_ac_pulse_parameters_for_channel(test.data['Iwe2'], test.data['Vwe2_ie'],
                                                                     test.data['Iie'], test,
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
            f"<p>Reference Document: {self.test_document}</p>",
            f"<p>Date/Time: {self.datetime}</p>",
            f"<p>EPG SN: FIXME </p>",
            f"<p>CPA: {self.cpa_sw}</p>",
            f"<p>DAQ Model #: {self.nidaq_product_type},  DAQ Tool #: {self.data_acq_system},"
            f"  Adapter Tool #: {self.daq_adapter}</p>",
            f"<p>Test Set Software: {self.test_software},  "
            f"Build ID: {self.test_software_build}</p>",
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
    def __init__(self, test):
        from datetime import datetime
        terr_l = 0.985  # Timing error based on 1.5% accurate clock
        terr_h = 1.015
        toff = 15  # Timing offset based on 15 uS timing resolution
        aerr_l = 0.95  # Amplitude error set to 5% for AC TECH10074.A12 | ZH  TECH1074.A3
        aerr_h = 1.05
        aoff = 25  # Amplitude offset of 25uA TECH1074.A12 | ZH 5uA TECH1074.A3
        dcerr = 1  # DC Current Error TECH1074.A11 | ZH=10 TECH1074.A10
        self.id = test.name()
        self.datetime = datetime.now()
        self.test_document = TestDoc.test_document
        self.test_software = TestDoc.test_software
        self.test_software_build = TestDoc.test_software_build
        self.cpa_sw = TestDoc.cpa_sw
        self.cpa_sw_build = TestDoc.test_software_build
        self.nidaq_product_type = test.testset.product_type
        self.data_acq_system = TestDoc.data_acq_system
        self.daq_adapter = TestDoc.daq_adapter
        self.epg_sn = "FIXME"
        self.peak_amp_ua = test.amp_ua
        self.rms_err_Vwe1 = test.data.eVwe1.std()
        self.rms_err_Vwe1 = test.data.eVwe1.std()
        self.vpp_err_Iwe1 = abs(test.data.eIwe1.max() - test.data.eIwe1.min())
        self.vpp_err_Iwe2 = abs(test.data.eIwe2.max() - test.data.eIwe2.min())
        self.vpp_err_Iie = abs(test.data.eIie.max() - test.data.eIie.min())
        self.vpp_err_Vwe1 = abs(test.data.eVwe1.max() - test.data.eVwe1.min())
        self.vpp_err_Vwe2 = abs(test.data.eVwe2.max() - test.data.eVwe2.min())
        self.iwe1_max = ['1', test.data.mIwe1.max(), test.data.rIwe1.max() * 0.95, test.data.rIwe1.max() * 1.05,
                         'TECH10074.A1', 'Iwe1-max', 'uA']
        self.iwe1_min = ['2', test.data.mIwe1.min(), test.data.rIwe1.min() * 0.95, test.data.rIwe1.min() * 1.05,
                         'TECH10074.A1', 'Iwe1-min', 'uA']
        self.rms_err_Iwe1 = ['3', test.data.eIwe1.std(), -10, 10, 'TECH10074.A10', 'Iwe1-rms-error', 'uA']
        self.iwe2_max = ['4', test.data.mIwe2.max(), test.data.rIwe2.max() * 0.95, test.data.rIwe2.max() * 1.05,
                         'TECH10074.A1', 'Iwe2-max', 'uA']
        self.iwe2_min = ['5', test.data.mIwe2.min(), test.data.rIwe2.min() * 0.95, test.data.rIwe2.min() * 1.05,
                         'TECH10074.A1', 'Iwe2-min', 'uA']
        self.rms_err_Iwe2 = ['6', test.data.eIwe2.std(), -10, +10, 'TECH10074.A10', 'Iwe1-rms-error', 'uA']
        self.iie_max = ['7', test.data.mIie.max(), test.data.rIie.max() - 10, test.data.rIie.max() + 10,
                        'TECH10074.A1', 'Iie-max', 'uA']
        self.iie_min = ['8', test.data.mIie.min(), test.data.rIie.min() - 10, test.data.rIie.min() + 10,
                        'TECH10074.A1', 'Iie-min', 'uA']
        self.rms_err_Iie = ['9', test.data.eIie.std(), -10.0, +10.0, 'TECH10074.A10', 'Iwe1-rms-error', 'uA']
        self.vwe1_max = ['10', test.data.mVwe1.max(), test.data.rVwe1.max() * 0.95, test.data.rVwe1.max() * 1.05,
                         'TECH10074.A6', 'Vwe1-max', 'mV']
        self.vwe1_min = ['11', test.data.mVwe1.min(), test.data.rVwe1.min() * 0.95, test.data.rVwe1.min() * 1.05,
                         'TECH10074.A6', 'Vwe1-min', 'mV']
        self.vwe2_max = ['12', test.data.mVwe2.max(), test.data.rVwe2.max() * 0.95, test.data.rVwe2.max() * 1.05,
                         'TECH10074.A6', 'Vwe1-max', 'mV']
        self.vwe2_min = ['13', test.data.mVwe2.min(), test.data.rVwe2.min() * 0.95, test.data.rVwe2.min() * 1.05,
                         'TECH10074.A6', 'Vwe1-min', 'mV']
        self.frequency = ['14', (test.sample_rate_per_channel / test.measured_samples_in_cycle) * 1000,
                          1000 / (12 * 0.99), 1000 / (12 * 1.01), 'TECH10074.A9', 'Frequency', 'mHz']
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
            f"<p>Reference Document: {self.test_document}</p>",
            f"<p>Date/Time: {self.datetime}</p>",
            f"<p>EPG SN: FIXME </p>",
            f"<p>CPA: {self.cpa_sw}</p>",
            f"<p>DAQ Model #: {self.nidaq_product_type},  DAQ Tool #: {self.data_acq_system},"
            f"  Adapter Tool #: {self.daq_adapter}</p>",
            f"<p>Test Set Software: {self.test_software},  "
            f"Build ID: {self.test_software_build}</p>",
            f"</body></html>"
        ]
        for line in lines:
            f.write(line)
        message = self.results.to_html()
        f.write(message)
        f.close()


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


def plot_numpy_array_charge(a, sr, plot_title='Current', y_axis_label='Current(uA)', ):
    """
    quick plot of numpy array
    """
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import LinearAxis, Range1d
    import numpy as np

    output_file(data_path() + 'SimpleNumpyPlot.html')
    p = figure(title=plot_title, y_axis_label=y_axis_label)
    x = np.linspace(0, len(a) - 1, len(a)) * (1 / sr)
    p.line(x, a, color='blue')

    b = np.cumsum(a)
    b = b / sr

    p.extra_y_ranges = {"Charge": Range1d(start=min(b), end=max(b))}
    p.add_layout(LinearAxis(y_range_name="Charge", axis_label='Charge (uC)'), 'right')

    p.line(x, b, y_range_name="Charge", color='red')
    show(p)


def list_nidaq_devices():
    print(list(system.System.local().devices))


def filter_60hz_notch(na, fs=666666, f0=60, q=30):
    from scipy import signal
    b, a = signal.iirnotch(f0, q, fs)
    y = signal.filtfilt(a, b, na)
    return y
