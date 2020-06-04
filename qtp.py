import ni
import pandas as pd
from datetime import datetime
import winsound
import xlrd
import xlwt


test_document = 'QP10330 rev.0'
tester_name = 'M. Faltys'


def run_test_script(filename, epg_sn='E00007', cpa_software='CSXXXXX', daq_tool='T0037-001',
                    daq_adapter='T0038-001', epg_adapter='TXXXXXX', test_software='CSXXXXX'):
    df = pd.read_excel(filename)  # TODO tell user to close files when they are open
    for index, row in df.iterrows():
        if pd.isnull(row['Test Name']):  # Test has not been run
            try:
                mode = row['Mode']
                amp_ua = row['amp_ua']
                pw_us = row['pw_us']
                ipi_us = row['ipi_us']
                rpr = row['rpr']
                frequency = row['frequency']
                bias = row['Bias']
                load = row['Load']
                if mode == 'AC':
                    winsound.Beep(2500, 200)
                    input(f'Configure EPG to AC Mode, Amplitude:{amp_ua}uA, Pulsewidth:{pw_us}uS, '
                          f'IPI:30uS, RPR:{rpr}x, Freq:{frequency}HzLoad Resistors:{load}ohms '
                          f'-- RETURN to CONTINUE')
                    test_repeat = 'R'
                    while test_repeat.lower():
                        try:
                            test_data, test_results, test_name = ni.test_ac(amp_ua, pw_us=pw_us, rpr=rpr,
                                                                            ipi_us=ipi_us, req_freq=frequency,
                                                                            load_resistance=load, cycles=1,
                                                                            epg_sn=ni.TestSetConfig.epg_sn)
                        except:
                            print('Something went wrong!!!')
                        winsound.Beep(1500, 300)
                        test_repeat = input('Input R to repeat or return to continue')

                elif mode == 'ZH':
                    winsound.Beep(2500, 200)
                    input(f'Configure EPG to ZeroHz Mode, Amplitude:{amp_ua}uA, Load Resistors:{load}ohms '
                          f'-- RETURN to CONTINUE')
                    print(f'Capture Started...')
                    test_repeat = 'R'
                    while test_repeat.lower():
                        try:
                            test_data, test_results, test_name = ni.test_zerohz(amp_ua, load_resistance=load,
                                                                     epg_sn=ni.TestSetConfig.epg_sn)
                        except:
                            print("Something went wrong!!!")
                        winsound.Beep(1500, 300)
                        test_repeat = input('Input R to repeat or return to continue')

                pass_fail_count = test_results.results['Pass/Fail'].value_counts().to_dict()
                print(f'Test Name {test_name},{datetime.now().strftime("%m%d%Y_%H_%M_%S")},{pass_fail_count}')
                df.loc[index, 'Test Name'] = test_name
                df.loc[index, 'Time'] = datetime.now().strftime("%m%d%Y_%H_%M_%S")
                df.loc[index, 'Pass_Fail'] = str(pass_fail_count)
                df.to_excel('Data/Temp.xlsx')
                print("Check if workbook was saved")

            except ValueError:
                print("ValueError occurred")
                return df
    return df


