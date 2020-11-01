import pmtest as pt
import pandas as pd
from datetime import datetime
import winsound
import sys


# Set-up Test Doc should be done here

def run_test_script(filename):
    try:
        df = pd.read_excel(filename, index_col=None)
    except Exception as e:
        print(e)
        sys.exit(1)

    winsound.Beep(2500, 200)
    input(f'About to calibrate test adapter, Turn stimulation off and RETURN to CONTINUE')
    test_repeat = 'r'
    testset = pt.TestSet('Dev1')
    test_results = []
    while test_repeat.lower() == 'r':
        testset.calibrate_daq_adapter_offset()
        winsound.Beep(1500, 300)
        test_repeat = input('Calibration complete, r to repeat, return to continue')

    for index, row in df.iterrows():
        if pd.isnull(row['Test Name']):  # Test has not been run
            testset.load_resistance = row['Load']
            if row['Mode'] == 'AC':
                # Pull data from spreadsheet and create a test class instance
                test = pt.AcTest(testset, row['amp_ua'], row['pw_us'], row['ipi_us'], row['rpr'], row['frequency'],
                                 epg_sn='E00007', load_resistance=row['Load'],
                                 sample_rate_per_channel=testset.sample_rate_per_channel)
                winsound.Beep(2500, 200)
                skip_request = input(f'Configure EPG to {test.name()}  -- RETURN to continue or s to skip')
                if skip_request == 's':
                    continue
                print(f'Capture Started...')
                test_repeat = 'r'
                while test_repeat.lower() == 'r':
                    try:
                        test.measure()
                        pt.plot_data_frame(test.data, test.name(), "Amplitude (uA/mV")
                        test_results = pt.TestResultAC(test)
                        test_results.print_report()

                    except Exception as e:
                        print(e)
                    winsound.Beep(1500, 300)
                    test_repeat = input('Input R to repeat or return to continue')

            elif row['Mode'] == 'ZH':
                test = pt.ZhTest(testset, row['amp_ua'], epg_sn='E007', load_resistance=row['Load'],
                                 sample_rate_per_channel=1000)
                winsound.Beep(2500, 200)
                skip_request = input(f'Configure EPG to {test.name()}  -- RETURN to continue or s to skip')
                if skip_request == 's':
                    continue
                print(f'Capture Started...')
                test_repeat = 'r'
                while test_repeat.lower() == 'r':
                    try:
                        test_data = test.measure()
                        pt.plot_data_frame(test.data, test.name(), "Amplitude (uA/mV")
                        test_results = pt.TestResultZH(test)
                        test_results.print_report()
                    except Exception as e:
                        print(e)
                    winsound.Beep(1500, 300)
                    test_repeat = input('Input R to repeat or return to continue')

            if test_results:  # Make sure
                pass_fail_count = test_results.results['Pass/Fail'].value_counts().to_dict()
                failed_params = test_results.results.loc[test_results.results['Pass/Fail'] == 'Fail']['Parameter']
                test_results = []
            else:
                pass_fail_count = "Test Analysis Failed"
            df.loc[index, 'Test Name'] = test.name()
            df.loc[index, 'Time'] = datetime.now().strftime("%m%d%Y_%H_%M_%S")
            df.loc[index, 'Pass_Fail'] = str(pass_fail_count)
            # print(failed_params.values)
            # df.loc[index, 'Failed_Params'] = failed_params.values
            df.to_excel('Data/Temp.xlsx', index=False)
            print(f'Test Name {test.name()},{datetime.now().strftime("%m%d%Y_%H_%M_%S")},{pass_fail_count}')

    testset = []  # Turn off test set
    return df
