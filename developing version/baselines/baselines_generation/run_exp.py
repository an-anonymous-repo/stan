from models.baselines import baseline1
from models.baselines import baseline2
from models.baseline3 import baseline3
from models.baseline4 import baseline4
from datetime import datetime
from datetime import timedelta
from utils.config_utils import recieve_cmd_config
import numpy as np
import pandas as pd
import argparse
import configparser
import gc
import sys
import os

# np.random.seed(131)
user_list = []
baseline_choice = 'baseline1'
day_number = 1
original_date = ''
real_gen = 'no'
saving_str = ''
working_folder = ''
sample_flag = 'False'
bins = []

def data_prepare(ip_str, sample_flag):
    source_data = working_folder + 'cleaned_data/expanded_day_1_%s.csv' % ip_str
    all_record = pd.read_csv(source_data)
    # print('pre', all_record.shape)
    if sample_flag == 'True':
        all_record = all_record.sample(frac=0.1, random_state=1)
    # print('after',all_record.shape)

    byt_train = np.reshape(all_record['byt'].values, (-1, 1))
    byt_log_train = np.log(byt_train)
    byt1_train = np.reshape(all_record['byt-1'].values, (-1, 1))
    byt1_log_train = np.log(byt1_train)

    time_delta_train = np.reshape(all_record['teDelta'].values, (-1, 1))
    sip = ip_str
    sip_train_pool = np.ravel(all_record['sa'].values)
    dip_train_pool = np.ravel(all_record['da'].values)

    teT_train = np.reshape(all_record['teT'].values, (-1, 1))
    df_col = all_record#['teT']
    

    return byt_log_train, time_delta_train, sip, sip_train_pool, dip_train_pool, byt1_log_train, teT_train, df_col

def model_prepare(original_date, sip, byt_log_train, time_delta_train, sip_train, dip_train, byt1_log_train=None, final_teT_train=None, teT_df_col=None):
    if baseline_choice == 'baseline1':
        meta_model = baseline1()
        meta_model.fit(original_date, sip, byt_log_train, time_delta_train, sip_train, dip_train, bins)
        return meta_model
    elif baseline_choice == 'baseline2':
        meta_model = baseline4()
        meta_model.fit(original_date, sip, byt_log_train, time_delta_train, sip_train, dip_train, byt1_log_train, final_teT_train, teT_df_col, bins)
        return meta_model
    elif baseline_choice == 'baseline3':
        meta_model = baseline3()
        meta_model.fit(original_date, sip, byt_log_train, time_delta_train, sip_train, dip_train, byt1_log_train, final_teT_train, teT_df_col, bins)
        return meta_model
    else:
        pass

def flush(gen_data):
    # write to a csv file
    import csv
    import os
    gen_file = working_folder+saving_str
    label = True
    if os.path.isfile(gen_file):
        label = False
    with open(gen_file, "a", newline="") as f:
        fieldnames = ['te', 'sa', 'da', 'byt', 'teDelta']
        writer = csv.writer(f)
        if label:
            writer.writerow(fieldnames)
        writer.writerows(gen_data)

def train_model(baseline_choice):
    starttime = datetime.now()
    final_byt_log_train = np.reshape(np.array([]), (-1, 1))
    final_time_delta_train = np.reshape(np.array([]), (-1, 1))
    final_sip = []
    final_sip_train = np.ravel(np.array([]))
    final_dip_train = np.ravel(np.array([]))
    final_byt1_log_train = np.reshape(np.array([]), (-1, 1))
    final_teT_train = np.reshape(np.array([]), (-1, 1))
    final_teT_df_col = None


    for deal_str in user_list:
        print('loading:' + deal_str)
        byt_log_train, time_delta_train, sip, sip_train_pool, dip_train_pool, byt1_log_train, teT_train, all_data_col = data_prepare(deal_str, sample_flag)

        final_byt_log_train = np.concatenate((final_byt_log_train, byt_log_train))
        final_time_delta_train = np.concatenate((final_time_delta_train, time_delta_train))
        final_sip.append(sip)
        final_sip_train = np.concatenate((final_sip_train, sip_train_pool))
        final_dip_train = np.concatenate((final_dip_train, dip_train_pool))

        final_byt1_log_train = np.concatenate((final_byt1_log_train, byt1_log_train))
        final_teT_train = np.concatenate((final_teT_train, teT_train))
        if final_teT_df_col is None:
            final_teT_df_col = all_data_col
        else:
            final_teT_df_col = final_teT_df_col.append(all_data_col) #np.concatenate((final_teT_df_col, teT_df_col)) #
        # print(len(final_byt_log_train))
        del byt_log_train, time_delta_train, sip, sip_train_pool, dip_train_pool, byt1_log_train, teT_train, all_data_col
        gc.collect()

    print("teT::", type(final_teT_df_col), final_teT_df_col.shape)
    model1 = model_prepare(original_date, final_sip, final_byt_log_train, final_time_delta_train, final_sip_train, final_dip_train, final_byt1_log_train, final_teT_train, final_teT_df_col)
    print(model1.likelihood)
    endtime = datetime.now()
    with open("exp_record.txt", "a") as myfile:
        myfile.write('train: ' + str(len(user_list)) + 'users,' + str(day_number) +'days,'+ baseline_choice
            + ' ==> train likelihood:' + str(model1.likelihood) + ' ==> outgoing prob:' + str(model1.outgoing_prob) 
            + ' ==> time:' + str((endtime-starttime).seconds/60) + 'mins\n')
    return model1

def gen_one(model1, for_whom):
    gen_data = []
    now_t = 0
    last_b = -1
    cnt = 0
    
    start_date_obj = datetime.strptime(original_date, '%Y-%m-%d %H:%M:%S')
    model1.reset_gen(for_whom)

    while True:
        dep_info = [now_t, last_b] if baseline_choice == 'baseline2' or baseline_choice == 'baseline3' else []
        gen_date_obj, gen_te, gen_sip, gen_dip, gen_byt, gen_te_delta = model1.generate_one(dep_info)
        gen_data.append([gen_te, gen_sip, gen_dip, gen_byt, gen_te_delta])
        now_t = int(str(gen_te)[11:13])
        last_b = gen_byt
        cnt += 1
        print(cnt, for_whom, gen_data[-1])

        date_spray = (gen_date_obj-start_date_obj).days
        # print('============', date_spray, type(date_spray), day_number)
        if date_spray >= day_number:
            if real_gen == 'yes':
                flush(gen_data[:-1])
                gen_data = [gen_data[-1]]
            break
    return cnt


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    # override the config with the command line
    # recieve_cmd_config(config['DEFAULT'])
    
    user_list = config['DEFAULT']['userlist'].split(',')
    baseline_choice = config['DEFAULT']['baseline']
    working_folder = config['DEFAULT']['working_folder']
    sample_flag = config['DEFAULT']['sample_from_raw_data']
    bins = list(map(float, config['DEFAULT']['bins'].split(',')))
    
    real_gen = config['GENERATE']['save_to_csv']
    gen_multi_user = config['GENERATE']['gen_multi_user']
    gen_users = config['GENERATE']['gen_users'].split(',')
    original_date = config['GENERATE']['original_date']
    day_number = int(config['GENERATE']['gen_daynumber'])
    
    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
    
    model1 = None
    if '-train' in sys.argv:
        # train the model
        model1 = train_model(baseline_choice)
        model1.save_the_model()
    
    if '-gen' in sys.argv:    
        starttime = datetime.now()
        # generate the data
        if model1 is None:
            if baseline_choice == 'baseline1':
                model1 = baseline1()
                model1.load_the_model()
            elif baseline_choice == 'baseline2':
                model1 = baseline4()
                model1.load_the_model()
            elif baseline_choice == 'baseline3':
                model1 = baseline3()
                model1.load_the_model()
            else:
                pass

        if gen_multi_user == 'True':
            print('generating multi users')
            storing_folder = 'mar2020_5times/%s_%sdays_folder' % (baseline_choice, day_number)
            if not os.path.exists('./data/gen_data/%s/' % storing_folder):
                os.makedirs('./data/gen_data/%s/' % storing_folder)
            print(len(gen_users), gen_users[0])
            # for i in range(len(gen_users)):
            tot_rows = 0
            piece = 0
            rd = 0
            while piece < 5:
                for i in range(10):
                    saving_str = "gen_data/%s/%d/%d_%s_%sdays_%s.csv" % (storing_folder, piece, rd, baseline_choice, day_number, user_list[i])
                    tot_rows += gen_one(model1, user_list[i])
                if tot_rows > 200000:
                    rd = 0
                    tot_rows = 0
                    piece += 1


        else:
            print('generating single user')
            saving_str = "gen_data/%s_%sdays.csv" % (baseline_choice, day_number)
            gen_one(model1, gen_users)

        endtime = datetime.now()
        with open("exp_record.txt", "a") as myfile:
            myfile.write('generating: ' + str(len(user_list)) + 'users,' + str(day_number) +'days,'+ baseline_choice
                + ' ==> time:' + str((endtime-starttime).seconds/60) + 'mins\n')
    