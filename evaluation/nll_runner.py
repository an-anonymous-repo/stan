"""
This file was copyed from pyleran2.distributions.parzen.py
Their license is BSD clause-3: https://github.com/lisa-lab/pylearn2/
"""
import numpy as np
import theano
import pandas as pd
from src.validation_utils import ip_split
from src.validation_utils import hms_to_second
T = theano.tensor
import time

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def log_mean_exp(a):
    """
    We need the log-likelihood, this calculates the logarithm
    of a Parzen window
    """
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def make_lpdf(mu, sigma):
    """
    Makes a Theano function that allows the evalution of a Parzen windows
    estimator (aka kernel density estimator) where the Kernel is a normal
    distribution with stddev sigma and with points at mu.
    Parameters
    -----------
    mu : numpy matrix
        Contains the data points over which this distribution is based.
    sigma : scalar
        The standard deviation of the normal distribution around each data \
        point.
    Returns
    -------
    lpdf : callable
        Estimator of the log of the probability density under a point.
    """
    x = T.matrix()
    mu = theano.shared(mu)
    
    a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma

    E = log_mean_exp(-0.5*(a**2).sum(2))

    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

    return theano.function([x], E - Z)


class ParzenWindows(object):
    """
    Parzen Window estimation and log-likelihood calculator.
    This is usually used to test generative models as follows:
    1 - Get 10k samples from the generative model
    2 - Contruct a ParzenWindows object with the samples from 1
    3 - Test the log-likelihood on the test set
    Parameters
    ----------
    samples : numpy matrix
        See description for make_lpdf
    sigma : scalar
        See description for make_lpdf
    """
    def __init__(self, samples, sigma):
        # just keeping these for debugging/examination, not needed
        self._samples = samples
        self._sigma = sigma

        self.lpdf = make_lpdf(samples, sigma)

    def get_ll(self, x, batch_size=10):
        """
        Evaluates the log likelihood of a set of datapoints with respect to the
        probability distribution.
        Parameters
        ----------
        x : numpy matrix
            The set of points for which you want to evaluate the log \
            likelihood.
        """
        inds = range(x.shape[0])
        n_batches = int(np.ceil(float(len(inds)) / batch_size))

        lls = []
        for i in range(n_batches):
            lls.extend(self.lpdf(x[inds[i::n_batches]]))

        return np.array(lls).mean()

def transformer(train_df, test_df):
    train_df = train_df[['te','pr','sa','sp','da','dp','byt','pkt','td']] # pkt
    test_df = test_df[['te','pr','sa','sp','da','dp','byt','pkt','td']] # pkt
    print(train_df.head(10))
    byt_max = np.log(max(train_df['byt'].max(), test_df['byt'].max()))
    pkt_max = np.log(max(train_df['pkt'].max(), test_df['pkt'].max()))
    td_max = max(train_df['td'].max(), test_df['td'].max())
    print(byt_max)

    def to_trans_space(df): 
        df['te'] = df['te'].apply(lambda x: hms_to_second(x))
        pr_dict = {'TCP':0,'UDP':1}
        df['pr'] = df['pr'].apply(lambda x: pr_dict[x] if x in pr_dict.keys() else 2).div(2.0)
        ip_split(df, 'sa', norm=True)
        df['sp'] = df['sp'].div(65535.0)
        ip_split(df, 'da', norm=True)
        df['dp'] = df['dp'].div(65535.0)
        df['byt'] = df['byt'].apply(lambda x: np.log(x+1)/byt_max)
        df['pkt'] = df['pkt'].apply(lambda x: np.log(x+1)/pkt_max)
        df['td'] = df['td'].div(td_max)
        return df
    train_df = to_trans_space(train_df)
    print(train_df.head())
    test_df = to_trans_space(test_df)
    return train_df, test_df

from sklearn.neighbors import KernelDensity

def kde_nll(real_data, gen_data):
    # arcnn
    #df_param = pd.read_csv('all_gen.csv')
    # wpgan
    #df_param = pd.read_csv('look_gen.csv')
    # baseline1
    #df_param = pd.read_csv('full_baseline1_day1.csv').sample(frac=0.1)
    # baseline2
    #df_param = pd.read_csv('full_baseline3_day1.csv').sample(frac=0.1)
    folder = './results/nll/2d_'
    print(folder+gen_data)
    print(folder+real_data)

    df_param = pd.read_csv(folder+gen_data)
    df_train = pd.read_csv(folder+real_data) 
    
    #df_1, df_2 = transformer(df_param, df_train)
    #df_1 = df_1.replace([np.inf, -np.inf], np.nan)
    #df_1 = df_1.dropna()
    #print(df_1['pkt'].max(), df_1['pkt'].min())

    #print(list(df_1.columns))
    #print(df_1.std())
    begintime = time.time()
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(df_param)
    endtime1 = time.time()
    row_ll = kde.score_samples(df_train)
    #pw_object = ParzenWindows(df_1, df_1.std())
    #row_ll = pw_object.get_ll(df_2)
    endtime2 = time.time()
    print('time:', endtime1-begintime, endtime2-endtime1)
    print('row_ll:', row_ll)
    print(gen_data, 'mean NLL:', sum(row_ll) / len(row_ll))
    with open('results/kde_record.txt', 'a') as f:
        print(row_ll, file=f)


def pr_mod(x):
    if x == 'TCP':
        return 0
    elif x == 'UDP':
        return 1
    else:
        return 2

def t_sne(data_name='real/day2_10user.csv', data_source_id=0):
    print('processing:', data_name)
    start_time = time.time()
    #input_vector = [[1,2,3...], [3,4,5....],....]
    df = pd.read_csv('postprocessed_data/%s' % data_name)
    #te,td,sa,da,sp,dp,pr,flg,fwd,stos,pkt,byt,lable,byt-1,teT,teS,teDelta
    #print(df.columns)
    df = df[['td','pkt', 'byt', 'sp','dp','pr']].dropna()
    df['pr'] = df['pr'].apply(pr_mod)
    #print(df[df.isna().any(axis=1)]) 
    tsne = TSNE(n_components=2)
    vector_2d = tsne.fit_transform(df)
    #print(vector_2d)
    #print(type(vector_2d))
    df_2d = pd.DataFrame(data=vector_2d, columns=['dim0', 'dim1'])
    df_2d['data_source'] = data_source_id
    #print(df_2d)
    df_2d.to_csv('./results/nll/2d_%s' % data_name.split('/')[-1], index=False)
    print("--- Finished in %s seconds ---" % (time.time() - start_time))

    #return
    #vector_2d = tsne.embedding_
    #x = map(lambda xy: xy[0], vector_2d)
    #y = map(lambda xy: xy[1], vector_2d)
    #c = map(lambda x: 1 if(x[0] > 1) else 0, result_vec)
    #plt.scatter(x, y,s=1)#, c=c)
    #plt.savefig('real_tsne2.png')

piece_id = 0
id_table = [
        'real/day2_10user.csv', 
        'bsl1/bsl1_piece%d.csv',
        'bsl2/bsl2_piece%d.csv',
        'wpgan/wpgan_piece%d.csv',
        'ctgan/ctgan_piece%d.csv',
        'arcnn_f/arcnn_f_piece%d.csv',
]

if __name__ == '__main__':
    for i in range(5, len(id_table)):
        nll = kde_nll(id_table[0].split('/')[-1], (id_table[i]%piece_id).split('/')[-1])
    #for i in range(len(id_table)):
    #    t_sne(id_table[i],i)
    #t_sne(id_table[5]%piece_id, 5)
