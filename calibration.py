import pandas as pd
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType

import numpy as np
import json

def write_calibration_file(rwe1=49.9, rwe2=49.9, rie=-49.9, rbat=0.28):
    """
    Writes calibration values to a JSON file with DICT structure - file not closed issue not found
    :param rwe1:
    :param rwe2:
    :param rie:
    :param rbat:
    :return:
    """
    sense_resistors = {'rwe1': rwe1, 'rwe2': rwe2, 'rie': rie, 'rbat': rbat}
    try:
        f = open('calibration.txt', 'w+')
        f.write(json.dumps(sense_resistors))
        f.close()
    except IOError:
        print("Calibration.txt not be opened")

def get_voltage_across_sense_resistors(channel):
    """
    Used by calibration utility to get voltage drops across sense resistors
    for a stimulation channel getting both the outgoing channel (WEx) and return (IE)
    :param channel: Either 'Iwe1' or 'Iwe2'
    :return: vwe, vie voltages across two sense resistors
    """

    sample_rate = 100000
    samples = int(sample_rate / 10)  # sample for 1/10 of a second

    physical_channel = f'{test_set.name}/ai0,{test_set.name}/ai4'  # Iwe1
    if channel == 'Iwe2':
        physical_channel = f'{test_set.name}/ai2,{test_set.name}/ai4'

    channel_gain = 0.5  # because 5mA x 50ohms = 0.25 and gain increments are 0.1, 0.2, 0.5, 1.0
    with nidaqmx.Task() as ai_task:
        ai_task.ai_channels.add_ai_voltage_chan(physical_channel, max_val=channel_gain, min_val=-channel_gain,
                                                terminal_config=TerminalConfiguration.DIFFERENTIAL)
        ai_task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=samples, sample_mode=AcquisitionType.FINITE)
        ai_task.start()
        di = np.array(ai_task.read(number_of_samples_per_channel=samples))

    vwe = di[0, :].mean() * 1000
    vie = di[1, :].mean() * 1000

    return vwe, vie


def calibrate_nidaq_using_6221_and_dvm(source='calibration.csv'):
    """
    Use a Keithley 6221 to a 2mA and then 15mA 0.5Hz looping through all current sense channels.
    a0 to a2 to a4 to a7. A DVM can iwe used to check the current level is the 6221 is not calibrated.
    Updates calibration.csv file
    :param source:
    :return:
    """
    rwe1 = None
    rwe2 = None
    rie = None
    rbat = None
    if source == 'calibration.csv':
        calibration = pd.read_csv("Calibration.csv")
        rwe1 = calibration.rwe1[0]
        rwe2 = calibration.rwe2[0]
        rie = calibration.rie[0]
        rbat = calibration.rbat[0]
    else:
        print('calibrate_nidaq_using_6221_and_dvm is not implemented')

    return rwe1, rwe2, rie, rbat


def calibrate_nidaq_using_dvm_and_psm(source='calibration.csv'):
    """
    Calibrate sense resistors or get calibration values. A
    The values of the sense resistors will be adjusted so that the DVM and NiDaq aligns
    :param source: either points to 'calibration.csv' file or requires user to 'recalibrate'
    :return: rwe1, rwe2, rie as derived from measurements
    """

    if source == 'calibration.csv':
        calibration = pd.read_csv("Calibration.csv")
        rwe1 = calibration.rwe1[0]
        rwe2 = calibration.rwe2[0]
        rie = calibration.rie[0]
        rbat = calibration.rbat[0]

    else:  # Recalibrate
        # Get data to build table
        input('Connect PSM to Nidaq and connect DVM current meter between Iwe1 and Iie(cr)')
        dvm1_neg = float(input('PSE Calibration Mode Set I1 Gain and measure negative current (-5.000)') or -5.0000)
        vwe1_neg, vie1_neg = get_voltage_across_sense_resistors('Iwe1')
        dvm1_pos = float(input('PSE Calibration Mode Set I1 Gain and measure positive current (+5.000)') or 5.0000)
        vwe1_pos, vie1_pos = get_voltage_across_sense_resistors('Iwe1')
        # DVM1_offset = float(input('PSE Calibration Mode Set I1 offset and measure offset current') or 0.0000)

        input('Connect PSM to Nidaq and connect DVM current meter between Iwe2 and Iie(cr)')
        dvm2_neg = float(input('PSE Calibration Mode Set I2 Gain and measure negative current (-5.000)') or -5.0000)
        vwe2_neg, vie2_neg = get_voltage_across_sense_resistors('Iwe2')
        dvm2_pos = float(input('PSE Calibration Mode Set I2 Gain and measure positive current (+5.000)') or 5.0000)
        vwe2_pos, vie2_pos = get_voltage_across_sense_resistors('Iwe2')
        # DVM_offset = float(input('PSE Calibration Mode Set I2 offset and measure offset current') or 0.0000)
        rwe1 = (vwe1_neg / dvm1_neg + vwe1_pos / dvm1_pos) / 2
        rwe2 = (vwe2_neg / dvm2_neg + vwe2_pos / dvm2_pos) / 2
        rie = (vie1_neg / dvm1_neg + vie1_pos / dvm1_pos + vie2_neg / dvm2_neg + vie2_pos / dvm2_pos) / 4
        rbat = .28

        calibration = pd.DataFrame({'rwe1': [rwe1], 'rwe2': [rwe2], 'rie': [rie]})
        calibration.to_csv("Calibration.csv", index=False)

    return rwe1, rwe2, rie, rbat