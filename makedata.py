# min-max scaling and Standardized [N(mu=0,sigma=1)]
# https://sebastianraschka.com/Articles/2014_about_feature_scaling.html

import pandas as pd
import numpy as np
import glob
import csv

memory_height = 5
memory_width = 16
gen_model = 2
train_user = 90
test_user = 1

def map_ip_str_to_int_list(ip_str, ipspace=None):
    ip_group = ip_str.split('.')
    rt = []
    pw = 1
    for i in list(reversed(range(len(ip_group)))):
        # print(i, int(ip_group[i]), pw)
        # rt += int(ip_group[i]) * pw
        rt.append(int(ip_group[i]))
        # if i>0:
        # pw *= 255
    # print(rt)
    if ipspace is not None:
        for i in range(len(rt)):
            rt[i] /= ipspace
    # print(rt)
    return rt

def port_number_interpreter(port_num, portspace):
    #sys_port = [21,22,25,53,80,110,137,138,139,143,443]
    #sys_len = len(sys_port)
    #rt = []

    rt = [port_num/portspace]
    #if port_num in sys_port:
    #    rt = [0] #, 1, 1.0*sys_port.index(port_num)/sys_len
    #else:
    #    rt = [port_num/portspace]#, 0, 0]
    return rt

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

def scaler_preprocess(train_df, test_df):
    all_df = pd.concat([train_df, test_df], axis=0,sort=False)
    df['log_byt'] = np.log(df['byt'])
    df['log_pkt'] = np.log(df['pkt'])

    min_max_scaler.fit(all_df)

def trainsfer(df,this_ip, file_name):
    buffer = [[0]*memory_width] * memory_height
    output_sample = []


def transform_1_df(df, this_ip, file_name, td_max, b_max):
    buffer = [[0]*memory_width] * memory_height
    output_sample = []

    df['log_byt'] = np.log(df['byt'])
    df['log_pkt'] = np.log(df['pkt'])
    if gen_model == 1:
        bytmax = 13.249034794106116 #20.12915933105231 # df['log_byt'].max()
        pktmax = 12.83
        tdmax = 363
        teTmax = 23 # df['teT'].max()
        teDeltamax = 222# 1336 # df['teDelta'].max()
        ipspace = 255
        portspace = 65535
    else:
        bytmax = 20.12915933105231 # df['log_byt'].max()
        pktmax = 12.83
        tdmax = 363
        teTmax = 23 # df['teT'].max()
        teDeltamax = 1336 # df['teDelta'].max()
        ipspace = 255
        portspace = 65535 
    # print('bytmax', bytmax, 'teTmax', teTmax, 'teDeltamax', teDeltamax, 'ipspace', ipspace)
    #byt_max = df['log_byt'].max()
    #byt_min = df['log_byt'].min()
    # make (n - 50) * 50 * 10 sample
    for index, row in df.iterrows():
        # each row: teT, delta_t, byt, in/out, tcp/udp/other, sa*4, da*4, sp_sig/sp_sys/sp_other, dp*3 
        line = [row['teT']/teTmax, row['teDelta']/td_max, row['log_byt']/b_max, row['log_pkt']/pktmax, row['td']/tdmax]
        # line = [row['teT']/teTmax, row['log_byt']/bytmax]
        # [out, in]
        sip_list = map_ip_str_to_int_list(row['sa'], ipspace)
        dip_list = map_ip_str_to_int_list(row['da'], ipspace)
        
        spo_list = port_number_interpreter(row['sp'], portspace)
        dpo_list = port_number_interpreter(row['dp'], portspace)

        if row['sa'] == this_ip:
            #line += sip_list + dip_list
            line += spo_list 
            line += dpo_list + dip_list 
            line += [1, 0]
        else:
            line += dpo_list
            line += spo_list + sip_list
            line += [0, 1]


        line_pr = []
        if row['pr'] == 'TCP':
            line_pr = [1, 0, 0]
        elif row['pr'] == 'UDP':
            line_pr = [0, 1, 0]
        else:
            line_pr = [0, 0, 1]
        line += line_pr

        
        #print(len(line))
        #print(line)
        #input()
        buffer.append(line)
        # if len(buffer)<51:
            # continue
        line_with_window = []
        for l in buffer[-memory_height-1:]:
            line_with_window += l
        # line_with_window = buffer[-50:]
        # print(line_with_window)
        # print(len(line_with_window))
        # input()
        output_sample.append(line_with_window)
    # print('linewithwindow', line_with_window)
    print(len(line_with_window), len(output_sample))

    with open('./input_data/day1_starter/%s.csv'%this_ip, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_sample[:1001])
    '''
    with open(file_name,'a') as f:
        writer = csv.writer(f)
        writer.writerows(output_sample)
    '''
    return len(output_sample)


print('========================start makedata==================')
if gen_model == 1:
    with open('train.csv','w') as f:
        pass
    source_data = 'expanded_day_1_42.219.145.151.csv'
    this_ip = source_data.split("_")[-1][:-4]
    print(this_ip)
    df = pd.read_csv(source_data)
    transform_1_df(df, this_ip, 'train.csv')
else:
    # df = pd.concat([pd.read_csv(f) for f in glob.glob('cleaned_data/*.csv')], ignore_index=True)
    #with open('input_data/all_train.csv','w') as f:
    #    pass
    #with open('input_data/day2_test.csv', 'w') as f:
    #    pass
    count = 0
    tot_train_line = 0
    tot_test_line = 0
    train_files = []
    test_files = []
    tedelta_max = -1
    log_byt_max = -1
    for two_set in ['input_data/train_set', 'input_data/day2_data']:
        for f in glob.glob(two_set+'/*.csv'):
            df = pd.read_csv(f)
            tedelta_max = max(tedelta_max, df['teDelta'].max())
            log_byt_max = max(log_byt_max, np.log(df['byt'].max()))

    count = 0
    for f in glob.glob('input_data/train_set/*.csv'):
        if count == train_user:
            break
        print('making train for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        train_files.append(f)
        tot_train_line += transform_1_df(df, this_ip, 'input_data/all_train.csv', tedelta_max, log_byt_max)
        count += 1
    '''
    count = 0
    for f in glob.glob('input_data/day2_data/*.csv'):
        if count == test_user:
            break
        print('making test for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        test_files.append(f)
        tot_test_line += transform_1_df(df, this_ip, 'input_data/day2_test.csv', tedelta_max, log_byt_max)
        count += 1
    '''
    print('tot_train_line:', tot_train_line, 'tot_test_line', tot_test_line)
    print('td_max', tedelta_max, 'b_max', log_byt_max)
    
    with open('input_data/makedata_record.txt', 'w') as f:
        f.write("train set\n"+"\n".join(train_files))
        f.write("\ntest set\n"+"\n".join(test_files))
        f.write('\ntot_train_line: '+ str(tot_train_line) + ' tot_test_line:' + str(tot_test_line) + '\n')
        f.write('\ntd_max: '+ str(tedelta_max) + ' b_max: ' + str(log_byt_max) + '\n')

#TODO: normalization to fixed range
#TODO: append the autoencoder module
