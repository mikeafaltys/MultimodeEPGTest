import ni
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
    test_repeat = 'R'
    while test_repeat.lower():
        ni.testset.calibrate_daq_adapter_offset()
        winsound.Beep(1500, 300)
        test_repeat = input('Input R to repeat or return to continue')

    for index, row in df.iterrows():
        if pd.isnull(row['Test Name']):  # Test has not been run
            ni.testset.load_resistance = row['Load']
            if row['Mode'] == 'AC':
                # Pull data from spreadsheet and create a test class instance
                test = ni.AcTest(row['amp_ua'], row['pw_us'], row['ipi_us'], row['rpr'], row['frequency'],
                                 epg_sn='E00007', load_resistance=row['Load'],
                                 sample_rate_per_channel=ni.testset.sample_rate_per_channel)
                winsound.Beep(2500, 200)
                input(f'Configure EPG to {test.name()}  -- RETURN to CONTINUE')
                print(f'Capture Started...')
                test_repeat = 'R'
                while test_repeat.lower():
                    try:
                        test_data = test.measure()
                        ni.plot_data_frame(test_data, test.name(), "Amplitude (uA/mV")
                        test_results = ni.TestResultAC(test, test_data)
                        test_results.print_report()

                    except Exception as e:
                        print(e)
                    winsound.Beep(1500, 300)
                    test_repeat = input('Input R to repeat or return to continue')

            elif row['Mode'] == 'ZH':
                test = ni.ZhTest(row['amp_ua'], epg_sn='E007', load_resistance=row['Load'],
                                 sample_rate_per_channel=1000)
                winsound.Beep(2500, 200)
                input(f'Configure EPG to {test.name()}  -- RETURN to CONTINUE')
                print(f'Capture Started...')
                test_repeat = 'R'
                while test_repeat.lower():
                    try:
                        test_data = test.measure()
                        ni.plot_data_frame(test_data, test.name(), "Amplitude (uA/mV")
                        test_results = ni.TestResultZH(test, test_data)
                        test_results.print_report()
                    except Exception as e:
                        print(e)
                    winsound.Beep(1500, 300)
                    test_repeat = input('Input R to repeat or return to continue')

            pass_fail_count = test_results.results['Pass/Fail'].value_counts().to_dict()
            print(f'Test Name {test.name()},{datetime.now().strftime("%m%d%Y_%H_%M_%S")},{pass_fail_count}')
            df.loc[index, 'Test Name'] = test.name()
            df.loc[index, 'Time'] = datetime.now().strftime("%m%d%Y_%H_%M_%S")
            df.loc[index, 'Pass_Fail'] = str(pass_fail_count)
            df.to_excel('Data/Temp.xlsx',index=False)
            print("Check if workbook was saved")

    return df
