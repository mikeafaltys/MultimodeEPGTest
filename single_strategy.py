
import pmtest as pm
import pandas as pd
from datetime import datetime
import winsound
import sys

# Set-up Test Doc should be done here
DEVICE_NAME = 'Dev4'
testdoc = pm.TestDoc()
ts = pm.TestSet(DEVICE_NAME)  # MAKE INSTANCE OF TestSet Class
testId = 1
testName = "Test 1"
mode = "ULF"
amp_ua = 1000
pw_us = 200
ipi_us = 30
rpr = 8
frequency = 50
bias = -42
load = 100
channel_pair = "E1_2"


ts.load = load
ts.ie_connect = "ON"
ts.load_hiZ = "OFF"
ts.channel_pair = channel_pair

if mode == "AC":
    test = pm.AcTest(ts, amp_ua=amp_ua, pw_us=pw_us, ipi_us=ipi_us, rpr=rpr, req_freq=frequency,
                     epg_sn='E00007', load=load,
                     sample_rate_per_channel=ts.sample_rate_per_channel)
    test_data = test.measure()
    pm.plot_data_frame(test_data, test.name(), "Amplitude (uA/mV")
    test_results = pm.TestResultAC(test)
    test_results.print_report()
elif mode == 'ULF':
    test = pm.ZhTest(ts, amp_ua=amp_ua, epg_sn='E007', load=load,
                     sample_rate_per_channel=1000)
    test_data = test.measure()
    pm.plot_data_frame(test_data, test.name(), "Amplitude (uA/mV")
    test_results = pm.TestResultZH(test)
    test_results.print_report()
# pass_fail_count = test_results.results['Pass/Fail'].value_counts().to_dict()
# print(f'Test Name {test.name()},{datetime.now().strftime("%m%d%Y_%H_%M_%S")},{pass_fail_count}')
