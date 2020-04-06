from sklearn import mixture
import pandas as pd
import numpy as np
from scipy import integrate

def pr_mod(x):
    if x == 'TCP':
        return 0
    elif x == 'UDP':
        return 1
    else:
        return 2

def anti_pr_mod(x):
    if x == 0:
        return 'TCP'
    elif x == 1:
        return 'UDP'
    else:
        return 'Others'

def ip_split(data, col_name):
    # new data frame with split value columns
    new = data[col_name].str.split(".", expand = True)
    for i in range(4):
        # making separate first name column from new data frame
        data[col_name+"_%d"%i]= new[i]

    # Dropping old Name columns
    data.drop(columns =[col_name], inplace = True)
    return data

def cal_likelihood(gmm_model, given_data, discrete=False):
    print(given_data)
    log_likeli = gmm_model.score(given_data)
    logprob = gmm_model.score_samples(given_data)
    add_likeli = sum(logprob)
    print('sys-calc likelihood:', log_likeli, add_likeli, add_likeli/len(logprob))
    likelihood = log_likeli

    gmm_pdf = lambda x: np.exp(gmm_model.score_samples(np.reshape([x], (-1, 1))))
    if discrete is False:
        bins = 304
        cats_byt = pd.cut(given_data.flatten(), bins, include_lowest=True)
    else:
        bins = [0,1,2]
        cats_byt = pd.cut(given_data.flatten(), bins, include_lowest=True)

    bin_likeli = 0
    dp = {}
    for id in range(0, len(given_data)):
        id_byt_interval = cats_byt[id]
        if id_byt_interval in dp:
            p_b = dp[id_byt_interval]
        else:
            p_b, _est_error = np.log(integrate.quad(gmm_pdf, id_byt_interval.left, id_byt_interval.right))
            dp[id_byt_interval] = p_b
        bin_likeli += p_b
    print(bin_likeli)
    input()

def complete_gmm(piece_i):
    df_train = pd.read_csv('90user_trainreal.csv')
    df_baseline1 = pd.read_csv('baseline1_1days_folder/bsl1_piece%d.csv'%piece_i)
    df_baseline3 = pd.read_csv('baseline2_1days_folder/bsl2_piece%d.csv'%piece_i)
    
    df_train = ip_split(df_train,'sa')
    df_train = ip_split(df_train,'da')
    df_train['pr'] = df_train['pr'].apply(pr_mod)
    df_train['pkt'] = np.log(df_train['pkt'])

    #to_compelete = ['td','sp','dp','pr','pkt','sa_0','sa_1', 'sa_2', 'sa_3', 'da_0', 'da_1', 'da_2', 'da_3']
    to_compelete = ['pkt']
    len_b1 = len(df_baseline1.index)
    len_b3 = len(df_baseline3.index)
    
    NLL = []
    for col in to_compelete:
        print('doing', col)
        col_train = np.array(df_train[col])
        col_train = np.expand_dims(col_train, axis=1)
        col_gmm_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(col_train)
        log_likeli = col_gmm_model.score(col_train)
        NLL.append(log_likeli)
        if col == 'pr':
            cal_likelihood(col_gmm_model, col_train, True)
        else:
            cal_likelihood(col_gmm_model, col_train, False)
        continue
        gen_1, _ = col_gmm_model.sample(len_b1)
        df_baseline1[col] = gen_1
        df_baseline1[col] = df_baseline1[col].astype(int)
        gen_3, _ = col_gmm_model.sample(len_b3)
        df_baseline3[col] = gen_3
        df_baseline3[col] = df_baseline3[col].astype(int)
    df_baseline1['pr'] = df_baseline1['pr'].apply(anti_pr_mod)
    df_baseline3['pr'] = df_baseline3['pr'].apply(anti_pr_mod)
    with open('attr_likelihood.txt','a') as f:
        print(NLL, file=f)
    print(NLL)
    print('nll', sum(NLL))
    #def export_func(df,name):
        
    #    df = df.replace([np.inf, -np.inf], np.nan)
    #    df = df.dropna()
    #    df.to_csv(name,index=False)
    #export_func(df_baseline1, '../../evaluation/postprocessed_data/bsl1/bsl1_piece%d.csv'%piece_i)
    #export_func(df_baseline3, '../../evaluation/postprocessed_data/bsl2/bsl2_piece%d.csv'%piece_i)
     

if __name__ == "__main__":
    for i in range(5):
        complete_gmm(i)   
