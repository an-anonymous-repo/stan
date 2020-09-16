import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score
import configparser
from utils.config_utils import recieve_cmd_config
from utils.plot_utils import boxplot
from utils.plot_utils import temporal_lineplot
from utils.plot_utils import distribution_lineplot
from utils.plot_utils import distribution_lineplot_in_one
from models.baselines import baseline1
from utils.distribution_utils import get_distribution_with_laplace_smoothing
from utils.distribution_utils import get_distribution
import glob

def KLc(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def loglikelihood(p):
    meta_model = baseline1()
    meta_model.load_params(meta_model.byt_model)

    logprob, responsibilities = meta_model.byt_model.score_samples(p)
    pdf = np.exp(logprob)
    return sum(logprob)

def cross_validation_preparation(data):
    k_fold = 5
    csvfile = open('import_1458922827.csv', 'r').readlines()
    filename = 1
    len_rows = int(len(csvfile) / float(k_fold) * k_fold)
    block_rows = len_rows / k_fold
    for i in range(k_fold):
        if i % block_rows == 0:
            open(str(filename) + 'import_1458922827.csv', 'w').writelines(csvfile[i:i+block_rows])
            filename += 1
    # build 5 model
    for i in range(k_fold):
        # open test data and test loglikelihood
        test_data = None
        loglikelihood(test_data)

working_folder = ''

def real_vali_KL(rawdata, gendata, bins):
    values1 = list(np.log(rawdata.byt))
    values2 = list(np.log(gendata.byt))

    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    cats2 = pd.cut(values2, bins)
    pr2 = list(cats2.value_counts())

    pk = get_distribution_with_laplace_smoothing(pr1)
    qk = get_distribution_with_laplace_smoothing(pr2)

    # If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0).
    # If qk is not None, then compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=0).
    KL = scipy.stats.entropy(pk, qk)

    # myKL = KLc(pk, qk)
    # NMI = normalized_mutual_info_score(pr2,pr2)

    return KL

def draw_3_distribution(name, rawdata, gen1data, gen2data, bins):
    import matplotlib.pyplot as plt
    values1 = list(np.log(rawdata.byt))
    values2 = list(np.log(gen1data.byt))
    values3 = list(np.log(gen2data.byt))

    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    cats2 = pd.cut(values2, bins)
    pr2 = list(cats2.value_counts())
    cats3 = pd.cut(values3, bins)
    pr3 = list(cats3.value_counts())

    pk = get_distribution(pr1)
    qk = get_distribution(pr2)
    q2k = get_distribution(pr3)
    
    _, ax1 = plt.subplots()
    ax1.plot(range(len(bins)-1), pk, label='real')
    ax1.plot(range(len(bins)-1), qk, label='baseline1')
    ax1.plot(range(len(bins)-1), q2k, label='baseline2')

    ax1.set_title(name)
    ax1.set_xlabel('bin')
    ax1.set_ylabel('pr')
    ax1.legend(loc=0, ncol=2)
    ax1.set(ylim=(0, 0.20))
    plt.savefig('figures/611/post/%s.png' % name)
    plt.close()

def gd(pd_data):
    values1 = list(np.log(pd_data.byt))
    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    # print(pr1)
    pk = get_distribution(pr1)
    return pk

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    x = x[:100000]
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

# def show_autocorrelation():
#     source_folder = './data/raw_data/'
#     summ = []
#     for f in glob.glob(source_folder+'*.csv'):
#         source_record = pd.read_csv(f)
#         values1 = np.log(source_record.byt).to_numpy()
#         e_a = estimated_autocorrelation(values1)
#         print(f, "estimated_autocorrelation", e_a, len(e_a))
#         summ.append(e_a[1])
#     print(sum(e_a)/len(e_a))

def show_autocorrelation():
    from statsmodels.tsa import stattools
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt

    source_folder = './data/raw_data/'
    rst = []
    for f in glob.glob(source_folder+'*.csv'):
        user_name = f.split('_')[-1][:-4]
        print("dealing with", user_name)

        source_record = pd.read_csv(f)
        byt_log_value = np.log(source_record.byt).to_numpy()
        np.log(source_record.byt).plot()
        plt.title("raw byt sequence of %s" % user_name)
        # plt.show()
        plt.savefig("auto-corr/%s_raw.png" % user_name)
        plt.close()
        acf = stattools.acf(byt_log_value)
        # print('acf:', acf)
        plot_acf(byt_log_value,use_vlines=True,lags=60,title="Autocorrelation for %s" % user_name)
        # plt.show()
        plt.savefig("auto-corr/%s_acf.png" % user_name)
        plt.close()
        pacf=stattools.pacf(byt_log_value)
        plot_pacf(byt_log_value,use_vlines=True,lags=60,title="Partial Autocorrelation for %s" % user_name)
        # plt.show()
        plt.savefig("auto-corr/%s_pacf.png" % user_name)
        plt.close()
        rst.append(','.join([user_name, str(acf[1]), '\n']))

    with open("afc-1.csv", "w") as myfile:
        # myfile.write(','.join(map(str, pk))+'\n')
        myfile.writelines(rst)

def count_tcp_udp():
    source_folder = './data/raw_data/'
    tcp_alluser = 0
    udp_alluser = 0
    all_alluser = 0
    all_keys = {}
    for f in glob.glob(source_folder+'*.csv'):
        user_name = f.split('_')[-1][:-4]
        print("dealing with", user_name)

        source_record = pd.read_csv(f)
        all_traffic = len(source_record.index)
        tcp_traffic = len(source_record[source_record['pr'] == 'TCP'].index)
        udp_traffic = len(source_record[source_record['pr'] == 'UDP'].index)
        print("all%d, tcp%f, udp%f" % (all_traffic, tcp_traffic/all_traffic, udp_traffic/all_traffic))
        all_protocol = source_record['pr'].unique()
        for i in all_protocol:
            all_keys[i] = 1

        tcp_alluser += tcp_traffic
        udp_alluser += udp_traffic
        all_alluser += all_traffic
    print("=============all in all============")
    print("all%d, tcp%f, udp%f" % (all_alluser, tcp_alluser/all_alluser, udp_alluser/all_alluser))
    print(all_keys.keys())

        

def process_distribution(bins):
    source_folder = './data/test_raw_data/'
    # target1_folder = './data/gen_data/5times_baseline1/5times%d_baseline1_1days_folder/' % 1
    # target2_folder = './data/gen_data/5times_baseline3/5times%d_baseline3_1days_folder/' % 1
    # NN_folder = './data/gen_data/NNmodel/'
    # source_folder = './data/mock_test/real/'
    target1_folder = './data/mock_test/b1/'
    target2_folder = './data/mock_test/b3/'
    # NN_folder = './data/mock_test/NNmodel813/'
    # NN_folder = './data/mock_test/watching/watching/bn_20user/'
    NN_folder = './data/mock_test/watching/watching/bn_90user/'


    print("processing dist of real")
    source_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)
    pk = gd(source_record)
    with open("./data/distribution/real_dist.txt", "w") as myfile:
        myfile.write(','.join(map(str, pk))+'\n')
    for t_hour in range(24):   
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        source_chunk = source_record[source_record['te'].str.contains(' '+str_hour+':')]
        pk = gd(source_chunk)
        with open("./data/distribution/real_dist_h%d.txt"%t_hour, "w") as myfile:
            myfile.write(','.join(map(str, pk))+'\n')

    print("processing dist of NN")
    NN_record = pd.concat([pd.read_csv(f) for f in glob.glob(NN_folder+'*.csv')], ignore_index = True)
    pk = gd(NN_record)
    with open("./data/distribution/NN_dist.txt", "w") as myfile:
        myfile.write(','.join(map(str, pk))+'\n')
    for t_hour in range(24):   
        # str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        NN_chunk = NN_record[NN_record['te'] == t_hour]
        print(t_hour)
        pk = gd(NN_chunk)
        with open("./data/distribution/NN_dist_h%d.txt"%t_hour, "w") as myfile:
            myfile.write(','.join(map(str, pk))+'\n')

    print("processing dist of b1")
    # for i in range(1, 6):
    target1_record = pd.concat([pd.read_csv(f) for f in glob.glob(target1_folder+'*.csv')], ignore_index = True)
    pk = gd(target1_record)
    with open("./data/distribution/baseline1_dist.txt", "w") as myfile:
        myfile.write(','.join(map(str, pk))+'\n')
    for t_hour in range(24):   
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        target1_chunk = target1_record[target1_record['te'].str.contains(' '+str_hour+':')]
        pk = gd(target1_chunk)
        with open("./data/distribution/baseline1_dist_h%d.txt"%t_hour, "w") as myfile:
            myfile.write(','.join(map(str, pk))+'\n')
    
    print("processing dist of b3")
    # for i in range(1, 6):
    target2_record = pd.concat([pd.read_csv(f) for f in glob.glob(target2_folder+'*.csv')], ignore_index = True)
    pk = gd(target2_record)
    with open("./data/distribution/baseline3_dist.txt", "w") as myfile:
        myfile.write(','.join(map(str, pk))+'\n')
    for t_hour in range(24):   
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        target2_chunk = target2_record[target2_record['te'].str.contains(' '+str_hour+':')]
        pk = gd(target2_chunk)
        with open("./data/distribution/baseline3_dist_h%d.txt"%t_hour, "w") as myfile:
            myfile.write(','.join(map(str, pk))+'\n')

def real_vali_JS(pk, qk, bins):
    avgk = [(x+y)/2 for x,y in zip(pk, qk)]

    JS = (scipy.stats.entropy(pk, avgk) + scipy.stats.entropy(qk, avgk)) / 2

    return JS

def vali_one(raw_ip_str, gen_ip_str, bins):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)

    KL = real_vali_JS(rawdata, gendata, bins)

    print(','.join([raw_ip_str, gen_ip_str, str(KL)]))
    return KL

def show_distribution(target_data, bins):
    values1 = list(np.log(target_data.byt))
    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    
    pk = get_distribution_with_laplace_smoothing(pr1)
    # print('===', len(pk))
    return pk

def show_conditioned_distribution(bins):
    target_folder = './data/raw_data/'    
    # target_folder = './data/nov_data/'

    # target_folder = './data/gen_data/baseline2_1days_folder/'
    target_folder = './data/mock_test/b3/'
    # target_folder = './data/gen_data/sample_baseline2_1days_folder/'
    # target_folder = './data/gen_data/argmax_baseline2_1days_folder/'
    
    target_record = pd.concat([pd.read_csv(f) for f in glob.glob(target_folder+'*.csv')], ignore_index = True)

    x_data = bins[1:]
    y_data = []
    traffic = []

    for t_hour in range(24):
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        target_chunk = target_record[target_record['te'].str.contains(' '+str_hour+':')]
        # target_chunk = target_record[target_record['te'] == t_hour]
        traffic.append(len(target_chunk.index))
        y_data.append(show_distribution(target_chunk, bins))
    
    print(traffic)
    distribution_lineplot(x_data, y_data, x_label='bins', y_label='probability', title='conditioned distribution')
    distribution_lineplot_in_one(x_data, y_data, x_label='bins', y_label='probability', title='conditioned distribution')

def vali_hourly(raw_ip_str, gen_ip_str, bins):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)
    
    rtn = []
    for T in range(24):
        hour_str = '00'
        if T<10:
            hour_str = hour_str[:-1] + str(T)[0]
        else:
            hour_str = str(T)
        # print('checking:', T, hour_str)
        raw_chunk = rawdata[rawdata['te'].str.contains(' '+hour_str+':')]
        gen_chunk = gendata[gendata['te'].str.contains(' '+hour_str+':')]
        # KL = real_vali_KL(raw_chunk, gen_chunk, bins)
        JS = real_vali_JS(raw_chunk, gen_chunk, bins)
        print(','.join(['For %d hour' % T, raw_ip_str, gen_ip_str, str(JS)]))
        rtn.append(JS)

    return rtn

def avg_distribution(f_name):
    ll = []
    with open('./data/distribution/%s.txt' % f_name) as fin:
        for line in fin:
            li = line.split(',')
            li = [float(i) for i in li]
            ll.append(li)
    # from __future__ import division
    
    print('avging:', f_name, len(ll), len(ll[0]))
    avg_dist = [sum(e)/len(e) for e in zip(*ll)]
    return avg_dist

def vali_as_a_whole(bins):
    avg_real_dist = avg_distribution('real_dist')
    avg_bsl1_dist = avg_distribution('baseline1_dist')
    avg_bsl2_dist = avg_distribution('baseline3_dist')
    avg_nn_dist = avg_distribution('NN_dist')

    print("real vs baseline1:", real_vali_JS(avg_real_dist, avg_bsl1_dist, bins))
    print("real vs baseline3:", real_vali_JS(avg_real_dist, avg_bsl2_dist, bins))
    print("real vs nn:", real_vali_JS(avg_real_dist, avg_nn_dist, bins))
    # draw_3_distribution('whole', source_record, target1_record, target2_record, bins)

    x_data = ['JS(raw|baseline1)', 'JS(raw|baseline3)', 'JS(raw|nn)']
    y_data = [[], [], []]

    for t_hour in range(24):   
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        # source_chunk = source_record[source_record['te'].str.contains(' '+str_hour+':')]
        # target1_chunk = target1_record[target1_record['te'].str.contains(' '+str_hour+':')]
        # target2_chunk = target2_record[target2_record['te'].str.contains(' '+str_hour+':')]
        # draw_3_distribution(str(t_hour), source_chunk, target1_chunk, target2_chunk, bins)
        avg_real_dist = avg_distribution('real_dist_h%d'%t_hour)
        avg_bsl1_dist = avg_distribution('baseline1_dist_h%d'%t_hour)
        avg_bsl2_dist = avg_distribution('baseline3_dist_h%d'%t_hour)
        avg_nn_dist = avg_distribution('NN_dist_h%d'%t_hour)

        y_data[0].append(real_vali_JS(avg_real_dist, avg_bsl1_dist, bins))
        y_data[1].append(real_vali_JS(avg_real_dist, avg_bsl2_dist, bins))
        y_data[2].append(real_vali_JS(avg_real_dist, avg_nn_dist, bins))
    
    average1 = sum(y_data[0])/len(y_data[0])
    average2 = sum(y_data[1])/len(y_data[1])
    average3 = sum(y_data[2])/len(y_data[2])
    
    # my_formatted_list = [ '%.3f' % elem for elem in y_data[0] ]
    # print("baseline1", my_formatted_list)
    
    # my_formatted_list = [ '%.3f' % elem for elem in y_data[1] ]
    # print("baseline2", my_formatted_list)
    
    print("average hour-conditioned JS:\nJS(real||baseline1):%f\nJS(real||baseline3):%f\nJS(real||nn):%f\n" % (average1, average2,average3))
    # temporal_lineplot(x_data, y_data, x_label='hour', y_label='JS divergency', title='3 pairs JS divergency compare')


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    # override the config with the command line
    recieve_cmd_config(config['DEFAULT'])
    
    user_list = config['DEFAULT']['userlist'].split(',')
    working_folder = config['DEFAULT']['working_folder']
    baseline_choice = config['DEFAULT']['baseline']
    bins = list(map(float, config['DEFAULT']['bins'].split(',')))
    
    test_list = config['VALIDATE']['test_set'].split(',')

    if config['VALIDATE']['raw_compare'] == 'True':
        x_data = []
        y_data = []
        for ip2 in user_list:
            x_data.append(ip2)
            y_data_i = []
            for ip1 in user_list:
                if ip1 == ip2:
                    continue
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'raw_data/day_1_%s.csv' % ip2 
                y_data_i.append(vali_one(src_file, des_file, bins))
            y_data.append(y_data_i)
            
        boxplot(x_data, y_data, title='KL(99 other raw ips || 1 raw_ip)')
    
    if config['VALIDATE']['hour_compare'] == 'True':
        for ip2 in user_list:
            x_data = []
            y_data = []
            for h in range(24):
                x_data.append(h)
                y_data.append([])
            
            for ip1 in user_list: 
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'gen_data/%s_1days_folder/%s_1days_%s.csv' % (baseline_choice, baseline_choice, ip2)
                y_data_i = vali_hourly(src_file, des_file, bins)
                for h in range(24):
                    y_data[h].append(y_data_i[h])
            print(len(y_data), len(y_data[0]))
        
            boxplot(x_data, y_data, title='KL(100 raw ips || %s)' % ip2)

    if config['VALIDATE']['conditioned_whole'] == 'True':
        show_conditioned_distribution(bins)

    # show_autocorrelation()
    # count_tcp_udp()

    if config['VALIDATE']['hour_whole'] == 'True':
        process_distribution(bins)
        vali_as_a_whole(bins)
