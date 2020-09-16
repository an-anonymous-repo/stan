import os
import glob
import pandas as pd
import numpy as np


def real_select(which_day='day1'):
    #ten_users = ['42.219.145.151', '42.219.152.127', '42.219.152.238', '42.219.152.246', '42.219.153.113', '42.219.153.115', '42.219.153.140', '42.219.153.146', '42.219.153.154', '42.219.153.158']
    save_file = './postprocessed_data/real/%s_90user.csv' % which_day
    if os.path.exists(save_file):
        os.remove(save_file)
    count = 0
    for filepath in glob.iglob('./generated_data/real_%s/*.csv' % which_day):
        if True: #filepath.split('_')[-1][:-4] in ten_users:
            print(filepath)
            count += 1
            df = pd.read_csv(filepath)
            if not os.path.isfile(save_file):
                df.to_csv(save_file, header='column_names', index=False)
            else: # else it exists so append without writing the header
                df.to_csv(save_file, mode='a', header=False, index=False)
    print('appended:', count)

def arcnn_gen_process():
    arcnn_version = 'arcnn_f90'
    os.system("awk '(NR == 1) || (FNR > 1)' generated_data/%s/*.csv > postprocessed_data/%s/%s_all.csv"%(arcnn_version, arcnn_version,arcnn_version))
    df = pd.read_csv('postprocessed_data/%s/%s_all.csv'%(arcnn_version, arcnn_version))
    for i in range(5):
        df_ = df.sample(frac=1/5.0)
        df_.to_csv('postprocessed_data/%s/%s_piece%d.csv'%(arcnn_version, arcnn_version, i),index=False)
    #os.system("awk '(NR == 1) || (FNR > 1)' generated_data/arcnn_gendata/arcnn_piece%d/*.csv > postprocessed_data/arcnn/arcnn_piece%d.csv"%(piece_i, piece_i))

def bsl_gen_process(piece_i, which_bsl = 'bsl1'):
    directory = './postprocessed_data/%s' % which_bsl
    if not os.path.exists(directory):
        os.makedirs(directory)
    gen_file = './generated_data/%s_gendata/%s_piece%d.csv' % (which_bsl, which_bsl, piece_i)
    df = pd.read_csv(gen_file)
    df['sa'] = df[['sa_0', 'sa_1', 'sa_2','sa_3']].astype(int).astype(str).agg('.'.join, axis=1)
    df['da'] = df[['da_0', 'da_1', 'da_2','da_3']].astype(int).astype(str).agg('.'.join, axis=1)
    rmv_cols = ['sa_0', 'sa_1', 'sa_2','sa_3', 'da_0', 'da_1', 'da_2','da_3']
    for col in rmv_cols:
        del df[col]
    
    df.to_csv('%s/%s_piece%d.csv'%(directory, which_bsl, piece_i),index=False)

def wpgan_gen_process(piece_i):
    directory = './postprocessed_data/wpgan'
    if not os.path.exists(directory):
        os.makedirs(directory)
    gen_file = './generated_data/wpgan_gendata/wpgan_piece%d.csv' % piece_i
    post_file = '%s/wpgan_piece%d.csv' % (directory, piece_i)
    head = 'day,te,td,pr,sa,sp,da,dp,pkt,byt,flg,lable'
    
    os.system("(echo \"%s\" && cat %s) | col -b > %s"%(head, gen_file, post_file))

def ctgan_gen_process():
    directory = './postprocessed_data/ctgan'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #byt,pkt,td,sp,dp,pr,sa_0,sa_1,sa_2,sa_3,da_0,da_1,da_2,da_3
    df = pd.read_csv('./generated_data/ctgan_gendata/ctgan_allpieces.csv')

    #print(df.head(5))
    df['byt'] = np.exp(df['byt']).astype(int)
    df['pkt'] = np.exp(df['pkt']).astype(int)
    df['sp'] = df['sp'].astype(int)
    df['dp'] = df['dp'].astype(int)
    df['sa'] = df[['sa_0', 'sa_1', 'sa_2','sa_3']].astype(int).astype(str).agg('.'.join, axis=1)
    df['da'] = df[['da_0', 'da_1', 'da_2','da_3']].astype(int).astype(str).agg('.'.join, axis=1)
    rmv_cols = ['sa_0', 'sa_1', 'sa_2','sa_3', 'da_0', 'da_1', 'da_2','da_3']
    for col in rmv_cols:
        del df[col]
    
    dfs_ = np.array_split(df, 5)
    count = 0
    for df_ in dfs_:
        df_.to_csv('%s/ctgan_piece%d.csv'%(directory, count),index=False)
        count += 1


if __name__ == "__main__":
    arcnn_gen_process()
    #for i in range(5):
    #    wpgan_gen_process(i)
    #ctgan_gen_process()
    #real_select(which_day='day1')
    #for i in range(5):
    #    bsl_gen_process(i, 'bsl1')
    #    bsl_gen_process(i, 'bsl2')
