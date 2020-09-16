import pandas as pd
import numpy as np
import csv
import sys
import os

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.nn_regressor import task2_regressor


# no_mix, mix_data
the_task = 'task2'
reduction_mode = '10-times'
#reduction_mode = 'percent'
arcnn_only = False
arcnn_version = 'arcnn_f90'
#exp_setting_list = [[False, arcnn_version], [True, ''], [False, 'bsl1'], [False, 'bsl2'], [False, 'wpgan'], [False, 'ctgan']]#, [False, 'arcnn']]#, [False, 'trainreal']]
#exp_setting_list = [[False, 'wpgan'], [False, 'ctgan']]#, [False, 'arcnn']]#, [False, 'trainreal']]
exp_setting_list = [[False, 'bsl2']]
#[False, 'ctgan']
#exp_setting_list = [[False, 'arcnn']]
#exp_setting_list = [[False, 'ctgan']]
result_record = 'results/tasks_record.txt' 

def pr_mod(x):
    if x == 'TCP':
        return 0
    elif x == 'UDP':
        return 1
    else:
        return 2

def lable_mod(x):
    if x == 'background':
        return 0
    elif x == 'blacklist':
        return 1
    else:
        return 2

def ip_split(data, col_name):
    # new data frame with split value columns
    new = data[col_name].str.split(".", expand = True)
    for i in range(4):
        # making separate first name column from new data frame
        data[col_name+"_%d"%i]= new[i]

    # Dropping old Name columns
    data.drop(columns =[col_name], inplace = True)
    return data

def flg_mod(x):
    if x == '.':
        return 0
    else:
        return 1

def flg_split(data):
    new = data['flg'].apply(lambda x: pd.Series(list(x)))
    
    for i in range(6):
        data['flg_%d'%i] = new[i]
        data['flg_%d'%i] = data['flg_%d'%i].apply(flg_mod)
    data.drop(columns=['flg'], inplace=True)
    return data

def fold_i_of_k(df_real,df_synth, synth_percent, no_synth, i, k):
    #======get the ith/k fold from the real data
    n = len(df_real)
    l_end = n*i//k
    r_end = n*(i+1)//k
    print('%d-th fold in %d in total: test is [%d~%d]' % (i, k, l_end, r_end))
    test_ = df_real.iloc[l_end:r_end]
    train_ = pd.concat([df_real.iloc[0:l_end], df_real.iloc[r_end:n]])
    
    #======mix synthetic data
    #num_of_synth_data = len(df_synth.index) * synth_percent // 100
    #num_of_real_data = len(df_real.index) - num_of_synth_data
    #df_train = pd.concat([df_train.iloc[:num_of_fake_data],df_test.iloc[:num_of_real_data]], axis=0)
    print('train_ len', len(train_.index), 'test_len', len(test_.index))
    if synth_percent <= 1:
        print('frac mode reduction,', synth_percent)
        if no_synth:
            train_ = train_.sample(frac=synth_percent)
            print('train_ amount', len(train_.index))
        else:
            synth_amount = int(len(train_.index) * (1-synth_percent))
            p_train = train_.sample(frac=synth_percent)
            p_synth = df_synth.sample(n=synth_amount)
            for col in p_synth.columns:
                if 'byt' not in col:
                    p_synth[col] = np.random.permutation(p_synth[col].values)
                    #p_synth['byt'] = np.random.permutation(p_synth['byt'].values)
            train_ = pd.concat([p_train, p_synth], axis=0,sort=False)
            print('train_ amount and synth_amount:', len(p_train.index), len(p_synth.index))
    else:
        num_of_real_data = len(train_.index) * (100-synth_percent) // 100
        num_of_synth_data = len(train_.index) * synth_percent // 100
        print('frac mode reduction, real num %d, synth num %d' % (num_of_real_data, num_of_synth_data))

        if no_synth:
            #train_ = train_.iloc[:num_of_real_data]
            train_ = train_.sample(n=num_of_real_data)
            print('train_ amount', len(train_.index))
        else:
            p_train = train_.sample(n=num_of_real_data)
            p_synth = df_synth.sample(n=num_of_synth_data)
            train_ = pd.concat([p_train, p_synth], axis=0, sort=False)
            print('train_ amount and synth_amount:', len(p_train.index), len(p_synth.index))
    
    return train_, test_

def preprocess(df):
    df.pr = df.pr.apply(pr_mod)
    #df.lable = df.lable.apply(lable_mod)
 
    #df = flg_split(df)
    if the_task == 'task2':
        #df = ip_split(df, 'sa')
        #df = ip_split(df, 'da')
        df['byt'] = df['byt'].apply(lambda x: np.log(x+1))
        #df['pkt'] = df['pkt'].apply(lambda x: np.log(x+1))
        df['nextbyt'] = df['byt'].shift(-1)
        df = df[:-1]
        rmv = ['pkt','sa', 'da','te','flg','day','fwd','stos','lable','byt-1','teT','teS','teDelta']
    elif the_task == 'task1' or 'task3':
        #df = ip_split(df, 'sa')
        #df = ip_split(df, 'da')
        #df['byt'] = df['byt'].apply(lambda x: np.log(x+1))
        #df['pkt'] = df['pkt'].apply(lambda x: np.log(x+1))
        rmv = ['sp','dp','sa','da','te','flg','day','fwd','stos','lable','byt-1','teT','teS','teDelta']
    else: #task3
        df = ip_split(df, 'sa')
        df = ip_split(df, 'da')
        rmv = ['sa','da', 'te','flg','day','fwd','stos','lable','byt-1','teT','teS','teDelta']
    for col in rmv:
        if col in df.columns:
            del df[col]
    #print(df.head())
    return df

def exclude_cur_row(df):
    rmv = ['byt', 'pkt', 'td','pr-5', 'pr-2', 'pr-3', 'pr-4']
    if the_task != 'task3':
        rmv.append('td-1')
    #rmv = ['byt', 'pkt', 'td']
    for col in rmv:
        if col in df.columns:
            del df[col]
    print('excluded df\n',df.head())
    return df

def expand_temperal(df, rollback_window):
    #print(df.head(10))
    for i in range(1,rollback_window):
        print('making:', i)
        for col in df.columns:
            if '-' in col or 'nextbyt' in col:
                continue
            df[col+'-%d'%i] = df[col].shift(i)
    df = df[rollback_window:]
    #print(df.head(10))
    #input()
    print('expanded cols:',len(list(df.columns)),list(df.columns))
    return df

def make_data(piece_i):
    data_list = ['bsl1', 'bsl2', 'wpgan', 'ctgan']
    data_name = ['./postprocessed_data/real/day1_90user.csv']
    for gen_data in data_list:
        data_name.append('./postprocessed_data/%s/%s_piece%d.csv'%(gen_data, gen_data, piece_i))
    
    #data_name = ['./postprocessed_data/%s/%s_all.csv'%(arcnn_version,arcnn_version)]
    data_name = ['./postprocessed_data/%s/%s_piece%d.csv'%(arcnn_version,arcnn_version,piece_i)]

    for data in data_name:
        df = pd.read_csv(data).dropna()
        df = preprocess(df)
        if the_task == 'task2':
            df = expand_temperal(df,5)
        elif the_task == 'task1':
            df = expand_temperal(df,2)
            df = exclude_cur_row(df)
        else: # task3
            df = expand_temperal(df,2)
            df = exclude_cur_row(df)
        
        if data.split('/')[-1] == 'day2_90user.csv':
            df = df.sample(frac=(1/10.0))
        elif 'arcnn_f90' in data.split('/')[-1]:
            df = df.sample(frac=(1/5.0))
        df = df.replace([np.inf, -np.inf], np.nan)
        print(len(df.index))
        df = df.dropna()
        print(len(df.index))
        save_folder = './_data/%d/'%piece_i
        if not os.path.exists(save_folder):
            os.makedirs(save_folder) 
        df.to_csv(save_folder+the_task+'_'+data.split('/')[-1], index=False)

def f1_validation(y_true, y_pred):
    # performance
    print(metrics.classification_report(y_true, y_pred))
    print(multilabel_confusion_matrix(y_true, y_pred))
    #print("F1 micro: %1.4f\n" % f1_score(y_test, y_predicted, average='micro'))
    print("F1 macro: %1.4f\n" % f1_score(y_true, y_pred, average='macro'))
    #print("F1 weighted: %1.4f\n" % f1_score(y_test, y_predicted, average='weighted'))
    #print("Accuracy: %1.4f" % (accuracy_score(y_test, y_predicted)))
    return f1_score(y_true, y_pred, average='macro')

def mixed_cross_validation(df_real, df_synth, synth_percent,no_synth=False):
    if no_synth and synth_percent == 100:
        return 0
    if no_synth and synth_percent == 0:
        return 0
    
    if the_task == 'task1':
        y_label = 'pr'
    elif the_task == 'task2':
        y_label = 'nextbyt'
    else:
        return -1

    if df_synth is not None: 
        print('synth_data amount:', len(df_synth.index))
    print(' real_data amount:', len(df_real.index))
    
    ret = []
    k_fold = 5
    importance = []
    for i_fold in range(k_fold):
        train_, test_ = fold_i_of_k(df_real, df_synth, synth_percent, no_synth, i_fold, k_fold)

        train_X = train_[train_.columns.difference([y_label])]
        train_y = train_[y_label]
        test_X = test_[test_.columns.difference([y_label])]
        test_y = test_[y_label]

        test_X = test_X[train_X.columns]
        #with open('result_record.txt', 'a') as f:
        #    print(test_X.columns, file=f)
        print('train_x cols:',len(list(train_X.columns)),list(train_X.columns))
        print('test_x cols:', test_X.columns)
        #print("train_lable_unique", train_.lable.unique())

        #clf = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0).fit(train_input, train_.pr)
        #clf = DecisionTreeClassifier(random_state=0).fit(train_input, train_.pr)
        #clf = make_model(train_X, train_y)
        if the_task == 'task1':
            #clf = RandomForestClassifier(random_state=0,n_estimators=100).fit(train_X, train_y)
            #input()
            clf = DecisionTreeClassifier(random_state=0).fit(train_X, train_y)
            print(clf.feature_importances_)
            importance.append(clf.feature_importances_)
        elif the_task == 'task2':
            #print(train_X.isnull().any())
            #print(train_y.isnull().any())
            #print('====what????')
            #print(train_X.max(), train_y.max())
            #print(train_X.min(), train_y.min())
            #clf = LinearRegression().fit(train_X, train_y)
            clf = task2_regressor().scale(train_X, test_X).fit(train_X, train_y)
        
        try:
            y_pred = clf.predict(test_X)
        except Exception as e:
            print(test_X)
            print(e) 
            input('?????')
        y_true = test_y

        if the_task == 'task1':
            lr_score = f1_validation(y_true, y_pred)
        elif the_task == 'task2':
            #y_true = np.exp(y_true) #y_true.apply(lambda x: np.log(x+1))
            #y_pred = np.exp(y_pred) #y_pred.apply(lambda x: np.log(x+1))
            lr_score = mean_squared_error(y_true, y_pred)
 
        print("="*20 + "Classification report for %d - fold validation" % i_fold)
        #ret.append(f1_validation(y_predicted, y_test))
        #lr_score = clf.score(test_X, test_y)
        print(lr_score)
        ret.append(lr_score)
    print('@@@@@@@@@',synth_percent, 'percent synthetic cross-validation:', ret, sum(ret)/len(ret))
    #with open('result_record.txt', 'a') as f:
    #    print(the_task, ' <-',synth_percent, 'percent synthetic 5-foldCV MSE score:', sum(ret)/len(ret), ret, file=f)    
    #    if the_task == 'task1':
    #        print('importance', importance[-1], file=f)
    return sum(ret)/len(ret)

def run(exp_setting, piece_i):
    summary = []
    mixed = exp_setting[1]
    if reduction_mode == 'percent':
        iter_list = [x*10 for x in range(11)]
    else:
        iter_list = [1, 0.1, 0.01, 0.001] #[1, 0.1, 0.01, 0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.000075, 0.00005,0.00001]
        iter_list = [1, 0.1, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0]
    #real_data_list = ['_90user_trainreal.csv', '_10user_real.csv', '_day2_90user.csv']
    #synth_data_list = ['_200k_wpgan.csv', '_gmm_arcnn.csv', '_90user_trainreal.csv', '_10user_arcnn.csv',
     #                   '_clf_10user_arcnn.csv', '_gmm_arcnn_dt_enhanced.csv', '_v1_enhanced_arcnn.csv',
      #                  '_full_baseline1_day1.csv', '_full_baseline3_day1.csv']
    real_data_list = ['day2_90user']
    synth_data_list = []
    real_data = './_data/0/%s_%s.csv'%(the_task, real_data_list[-1])
    synth_data = './_data/%d/'%piece_i
    synth_data += '%s_%s_piece%d.csv'%(the_task, mixed , piece_i)
    #synth_data += '%s_%s_all.csv'%(the_task, mixed)
    #if mixed == 'wpgan':
    #    synth_data = synth_data+'%s_%s_piece%d'%(the_task, 'wpgan', piece_i)
    #elif mixed == 'ctgan':
    #    synth_data = './'+the_task+'_90user_m_ctgan.csv'
    #elif mixed == 'arcnn':
    #    our_data = 1 if the_task == 'task2' else 6
    #    synth_data = './'+the_task+synth_data_list[our_data]
    #elif mixed == 'trainreal':
    #    synth_data = './'+the_task+synth_data_list[2]
    #elif mixed == 'baseline1':
    #    synth_data = './'+the_task+synth_data_list[7]
    #elif mixed == 'baseline3':
    #    synth_data = './'+the_task+synth_data_list[8]

    df_synth = pd.read_csv(synth_data).sample(frac=1) if exp_setting[0] is False else None
    df_real = pd.read_csv(real_data)
    #print('======>',np.where(np.isnan(df_synth)))
    #print(df_synth.isnull().any())
    
    print(df_real[:10])
    print('for real pearson coef\n', df_real.corr(method ='pearson'))
    #input()
    for i in iter_list:
        frac = i
        summary.append(str(mixed_cross_validation(df_real, df_synth, frac,no_synth=exp_setting[0])))
    exp_name = exp_setting[1] if exp_setting[0] is False else 'real_only'
    with open(result_record, 'a') as f:
        #print('no_synth is', exp_setting[0],',synth_data is', exp_setting[1], file=f)
        print(','.join([exp_name]+summary), file=f)

if __name__ == "__main__":
    if '-m' in sys.argv:
        for i in range(5):
            make_data(i)

    if '-en' in sys.argv:
        pr_enhanced()
     
    if '-run' in sys.argv:
        with open(result_record, 'a') as f:
            print(the_task+'=========================', file=f)
        for i in exp_setting_list:
            for j in range(5):
                run(i, j)
            if arcnn_only:
                break
        with open(result_record, 'a') as f:
            print('end===========================', file=f)
