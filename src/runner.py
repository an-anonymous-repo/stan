#NLL is: 2.73
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch
import csv
import time
import sys
import os
import random
import torch.distributions as tdist
import configparser
from nets.single_task_nets import SingleTaskNet
from net_cnn_pp.all_in_one_nn import NTCNN
from net_cnn_pp.dec_nn import NTCNN2# as NTCNN2
from net_cnn_pp.shallow_nn import shallow_ARCNN
import nets.single_task_nets as ncp
import glob
from random import randrange
from train_utils import int_to_bits
from train_utils import bits_to_int
from train_utils import make_one_hot
from sklearn.preprocessing import OneHotEncoder

# initialize config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
EPOCH = 3000
default_load_epoch = 591
memory_height = 5
traffic_feature_num = 16
# 0 is continuous
# 1 is discrete
# [teT, teDelta, log-byt, log-pkt, td, local_p_gmm, out_p_gmm, out_ip1, out_ip2, out_ip3, out_ip4] 0~10 
# [out/in]11~12; [tcp,udp,other]13~15; [sys/regis sp]5; [sys/regis dp]6;
# [sp-10bit]5; [dp-10bit]6 
gen_list = [1,2,3,4,5,6,7,8,9,10]
model_list = {}
optimizer_list = {}
scheduler_list = {}
# continuous
for idx in gen_list:
    model_list[idx] = SingleTaskNet(target_type='continuous')
port_way = 1670 # 1024+646
#discrete_dim = {7:256, 8:256, 9:256, 10:256, 11:2, 13:3, 20:256, 21:256, 22:256, 23:256, 24:1670, 25:1670}
discrete_dim = {11:2, 13:3, 20:256, 21:256, 22:256, 23:256, 24:1670, 25:1670}
# discrete (including one-hot)
#model_list[7] = SingleTaskNet(target_type='discrete', dim=256)
#model_list[8] = SingleTaskNet(target_type='discrete', dim=256)
#model_list[9] = SingleTaskNet(target_type='discrete', dim=256)
#model_list[10] = SingleTaskNet(target_type='discrete', dim=256)
model_list[11] = SingleTaskNet(target_type='discrete', dim=2)
model_list[13] = SingleTaskNet(target_type='discrete', dim=3)
# append
rely_attr = {20:5,21:5,22:6,23:6,24:5,25:6}
model_list[24] = SingleTaskNet(target_type='discrete', dim=1670)
model_list[25] = SingleTaskNet(target_type='discrete', dim=1670)

for key in sorted(model_list.keys()):
    model_lr = 1e-2 if key in discrete_dim.keys() else 1e-3
    optimizer_list[key] = optim.Adam(model_list[key].parameters(), lr=model_lr)
    scheduler_list[key] = optim.lr_scheduler.StepLR(optimizer_list[key],step_size=10,gamma=0.9)

def save_model(name, epoch, model):
    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    if not os.path.exists('./saved_model/model_%d'%name):
        os.makedirs('./saved_model/model_%d'%name)
    checkpoint = './saved_model/model_%d'%name + '/ep%d.pkl'%epoch
    torch.save(model.state_dict(), checkpoint)

def load_model(name, epoch, model):
    checkpoint = './saved_model/model_%d'%name + '/ep%d.pkl'%epoch
    model.load_state_dict(torch.load(checkpoint))


from torchsummary import summary
#summary(model_list[1].cuda(), (memory_height, traffic_feature_num))
#print(model_list[11])

#mdn_gen = model.mdn_cfgs['out_num']
bin_width = {
            'dtime': 1/1000,
            'byt': 0.0027,
            'ip' : 1/255,
            'port': 1/65536,
        }
#bin_width = model.mdn_cfgs['bin_width']
bin_width = 1/200.0
if torch.cuda.device_count() > 1:
    for key in sorted(model_list.keys()):
        model_list[key] = nn.DataParallel(model_list[key])
        model_list[key].to(device)

gen_feature_num = 1
train_batch_size = int(512)
conv_num = int(64/4)
window_pool = [1, 3, 3, 5]
gauss_num = 7

checkpoint_file = './saved_model/mixed_model-ep%d.pkl'
train_model = 2
if train_model == 1:
    bytmax, teDeltamax, train_file, test_file, gen_file, loss_file = \
        13.249034794106116, 222, 'train.csv', 'train.csv', 'gen.csv', 'small_loss.csv'
else:
    bytmax, teDeltamax, pktmax, tdmax, train_file, test_file, gen_file, loss_file = \
        20.12915933105231, 1430, 12.83, 363, 'input_data/enhanced_all_train.csv', 'input_data/day2_test.csv', 'train1_enhance_gen/all_gen_%s_%d.csv', 'all_loss.csv'
#1336

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width

    def __getitem__(self, index):
        single_image_label = np.asarray(self.data.iloc[index]).reshape(self.height,self.width).astype(np.float32)[-1]
        img_as_np = np.asarray(self.data.iloc[index]).reshape(self.height,self.width).astype(np.float32)[:-1]
        img_as_tensor = torch.from_numpy(img_as_np)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

def check_cuda():
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    else:
        print('CPU')



def test_validation(epoch, validation_loader): 
    #model.eval()
    start_time = time.time()

    model_list = ['dec%d'%i for i in range(dec_gen)] + ['mdn%d'%i for i in range(mdn_gen)] 
    md_loss = {}
    for idx in model_list:
        md_loss[idx] = []
    for step, (x, b_label) in enumerate(validation_loader):
        minibatch = x.view(-1, memory_height, traffic_feature_num)
        model.zero_grad()
        
        mdn_out_, dec_out_ = model(minibatch.to(device))
        col_offset = 1
        total_loss = 0
        for i in range(len(mdn_out_)):
            [pi, sigma, mu] = mdn_out_[i]
            tar = b_label.view(-1, traffic_feature_num)[:, col_offset+i].view(-1, 1)
            loss = ncp.mdn_loss(model, pi, sigma, mu, tar.to(device), bin_width)
            
            #if 'mdn%d'%i not in md_loss.keys():
            #    md_loss['mdn%d'%i] = []
            md_loss['mdn%d'%i].append(loss.data.tolist())
            if (torch.isnan(loss).any()):
                continue
            total_loss = loss if total_loss == 0 else total_loss + loss
        
        col_offset += len(mdn_out_)
        for i in range(len(dec_out_)): 
            pred = dec_out_[i]
            tar = b_label.view(-1, traffic_feature_num)[:, col_offset+i:col_offset+dec_out[i]]
            loss = ncp.dec_loss(pred, tar.to(device))
            #if 'dec%d'%i not in md_loss.keys():
            #    md_loss['dec%d'%i] = []
            md_loss['dec%d'%i].append(loss.data.tolist())
            if (torch.isnan(loss).any()):
                continue 
            total_loss = loss if total_loss == 0 else total_loss + loss

        print('Epoch:', epoch, ' Step:', step, 'len: ', len(md_loss.keys()), end=':')
        for key in sorted(md_loss.keys()):
            print(key, md_loss[key][-1], end=',')
        print()

    end_time = time.time()
    validation_curve_loss = [sum(md_loss[key])/len(md_loss[key]) for key in sorted(md_loss.keys())]
    
    temp = [epoch, end_time-start_time] + validation_curve_loss
    print('Epoch: ', epoch, '| test loss: ', validation_curve_loss, 'time cost as second:', end_time-start_time)

    with open('validation_'+loss_file,'a') as f:
        writer = csv.writer(f)
        writer.writerows([temp])
  
    model.train()

def train(load='False'):
    print('='*25+'start loading train data'+'='*25)
    if load == 'False':
        train_from_csv = CustomDatasetFromCSV(train_file, memory_height+1, traffic_feature_num)
        train_loader = torch.utils.data.DataLoader(dataset=train_from_csv, batch_size=train_batch_size, shuffle=True, \
                num_workers=16, pin_memory=True) 
        i = 0
        with open(loss_file,'w') as f:
            writer = csv.writer(f)
            writer.writerows([['epoch','time']+sorted(model_list.keys())])
    else:
        test_from_csv = CustomDatasetFromCSV(test_file, memory_height+1, traffic_feature_num) 
        vali_loader = torch.utils.data.DataLoader(dataset=test_from_csv, batch_size=train_batch_size*2, shuffle=True, \
                num_workers=16, pin_memory=True)
        i = load
        for idx in sorted(model_list.keys()):
            model_list[task_key].eval()

    print('='*25+'start training'+'='*25) 
    train_loss = []
    
    for epoch in range(i,EPOCH):
        start_time = time.time()

        md_loss = {}
        for idx in sorted(model_list.keys()):
            md_loss[idx] = []
        for step, (x, b_label) in enumerate(train_loader):
            minibatch = x.view(-1, memory_height, traffic_feature_num)
            for task_key in sorted(model_list.keys()):
                model = model_list[task_key] 
                out_ = model(minibatch.to(device))
                #print(task_key)
                
                if task_key >= 20: # append attributes
                    pred = out_
                    tar = b_label.view(-1, traffic_feature_num)[:, rely_attr[task_key]]
                    #tar_ = tar.cpu().detach().numpy()
                    #if task_key in [20, 21]:
                        #tar_ = (tar < 1024/65536).float().view(-1,1)
                        #print(tar_)
                        #temp_tar = make_one_hot(tar_)
                        #print(temp_tar)
                    #else: #[22, 23]
                        #torch.from_numpy(x).float()
                    if task_key in [20, 22]: # upper 8-bit
                        tar_ = np.rint((tar * 65536).cpu().data.numpy()).astype(int)
                        tar_2 = torch.from_numpy(tar_//256)
                        tar_2 = torch.clamp(tar_2, max=255)
                    elif task_key in [21, 23]: # [21, 23] lower 8-bit
                        tar_ = np.rint((tar * 65536).cpu().data.numpy()).astype(int)
                        tar_2 = torch.from_numpy(tar_%256)
                    else: # [24, 25]
                        #print(tar.size())
                        def get_category(x):
                            return (x-1024)//100+1024 if x >= 1024 else x
                        tar_ = np.rint((tar * 65536).cpu().data.numpy()).astype(int)
                        #print(tar_)
                        tar_ = np.array([get_category(xi) for xi in tar_])
                        #print(tar_)
                        tar_2 = torch.from_numpy(tar_)
                        #input()
                    #print(type(tar_), tar_)
                    #print(type(tar_2), tar_2.size(), tar_2)
                    loss = ncp.dec_loss(pred, tar_2.to(device))
                        #temp_tar = int_to_bits(tar_)
                        #print('mid', temp_tar)
                        #temp_tar = torch.from_numpy(temp_tar).float()
                        #print('22/23', temp_tar)
                        #print(pred)
                        #input()
                    #loss = ncp.bce_loss(pred, temp_tar.to(device))
                elif task_key <= 6:#model.target_type == 0:
                    [pi, sigma, mu] = out_
                    tar = b_label.view(-1, traffic_feature_num)[:, task_key].view(-1, 1)
                    loss = ncp.mdn_loss(model, pi, sigma, mu, tar.to(device), bin_width)
                elif task_key <= 10:
                    pred = out_
                    tar = b_label.view(-1, traffic_feature_num)[:, task_key].view(-1, 1)
                    tar_ = np.rint(tar * 255).long().squeeze(1)
                    loss = ncp.dec_loss(pred, tar_.to(device))

                else: # target_type == 1:
                    pred = out_
                    tar = b_label.view(-1, traffic_feature_num)[:, task_key:task_key+discrete_dim[task_key]]#.view(-1,1)
                    #print(pred)
                    #print(tar)
                    #print("loc", task_key, task_key+discrete_dim[task_key], discrete_dim[task_key])
                    #input()
                    
                    #one_hot_target = make_one_hot(tar, num_classes=pred.shape[1])
                    #print(task_key, len(pred), one_hot_target)
                    #loss = ncp.dec_loss(pred, one_hot_target.to(device))
                    #print(tar.size())
                    loss = ncp.dec_loss(pred, tar.to(device), is_onehot=True)
                    #print(loss)

                md_loss[task_key].append(loss.data.tolist())
                if epoch > 0: # start to train
                    if (torch.isnan(loss).any()):
                        #print(loss)
                        #input()
                        continue 
                    #print('do step')
                    optimizer_list[task_key].zero_grad()
                    loss.backward()
                    #nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer_list[task_key].step()

            print('Epoch:', epoch, ' Step:', step, 'len: ', len(md_loss.keys()), end=':')
            for key in sorted(md_loss.keys()):
                print(key, md_loss[key][-1], end=',')
            print()
        
        #input()
        for key in sorted(model_list.keys()):
            save_model(key, epoch,model_list[key])
            scheduler_list[key].step()
        #print('current lr', scheduler_list[1].state_dict()['param_groups'][0]['lr'])   

        # print logs
        # print("avgloss", train_avg_loss)
        train_avg_loss = [sum(md_loss[key])/len(md_loss[key]) for key in sorted(md_loss.keys())]
        end_time = time.time()
        temp = [epoch, end_time-start_time] + train_avg_loss # + validation_curve_loss 

        # print('runtime as second', end_time-start_time)
        print('Epoch: ', epoch, '| train loss: ', train_avg_loss, 'time cost as second:', end_time-start_time)
        with open(loss_file,'a') as f:
            writer = csv.writer(f)
            writer.writerows([temp])

def rectify(val, rect_low):
    if val < 0:
        val = (1/rect_low)
    if val > 1:
        val = (1/rect_low)
    return val

from sklearn.tree import DecisionTreeClassifier
def pr_enhancer():
    the_task = 'task1'
    real_data_list = ['_90user_trainreal.csv', '_10user_real.csv', '_day2_90user.csv']
    real_data = './input_data/dt_enhance_day1_90user.csv' #+the_task+real_data_list[0]

    df_real = pd.read_csv(real_data)

    y_label = 'pr'
    train_y = df_real[y_label]
    train_X = df_real[df_real.columns.difference([y_label])]
    #print(train_X.columns)
    #input()
    #test_X = test_X[train_X.columns]
    clf = DecisionTreeClassifier(random_state=0).fit(train_X, train_y)
    #pred_y = clf.predict(test_X)
    return clf

def pr_mod(x):
    if x == 'TCP':
        return 0
    elif x == 'UDP':
        return 1
    else:
        return 2

def enhance_pr(clf, last_row, this_row):
    #['te', 'teDelta', 'byt','pkt','td','sa','sp','da', 'dp', 'pr']
    #'byt-1', 'pkt-1', 'pr-1', 'td-1']
    #td-1,pr-1,pkt-1,byt-1
    #test_X_row = [[last_row[2], last_row[3], pr_mod(last_row[-1]), last_row[4]]]
    test_X_row = [[last_row[4], pr_mod(last_row[-1]), last_row[3], last_row[2]]]
    pred_y = clf.predict(test_X_row)
    print('pr', pred_y)
    return pred_y[0]

pr_enhance = False

def test(this_ip):
    print('='*25+'start loading test data'+'='*25)
    starter_file = './input_data/day1_starter/%s.csv'%this_ip
    test_from_csv = CustomDatasetFromCSV(starter_file, memory_height+1, traffic_feature_num)
    test_loader = torch.utils.data.DataLoader(dataset=test_from_csv, batch_size=1000, shuffle=False)
    if pr_enhance:
        clf = pr_enhancer()
    print('='*25+'start testing'+'='*25)
    marginal = None
    generated_rows = []
    i = 0
    #model.eval()
    start_row = randrange(1000)
    for step, (x, b_label) in enumerate(test_loader):
        # print('testing:::::::', x.size())
        marginal = x.view(-1, memory_height*traffic_feature_num)[start_row]
        break
        #minibatch = x.view(-1, memory_height, traffic_feature_num)
        ## print(minibatch)
        #model.zero_grad()
        #pi, sigma, mu = model(minibatch.to(device))
        #samples = mdn.sample(pi, sigma, mu)
        #samples[-1][0] = torch.round(minibatch[0][-1][0] * 23)
        ## print('minihour', minibatch[-1][0][0])
        #new_row = deal_with_output(samples)
        #generated_rows.append(new_row)
        #i+=1
        #print('%dth row'%i, new_row)
    print("marginal row", marginal)
    tot_time = 0
    row_num = 0
    generated_rows = []
    print('='*25+'testing'+'='*25)
    # print('marginal testing', marginal.size(), marginal.view(-1, 50, 10))
    while tot_time < 86400:
        minibatch = marginal.view(-1, memory_height, traffic_feature_num)
        samples = []
        outputs = []

        for task_key in sorted(model_list.keys()):
            model = model_list[task_key]
            out_ = model(minibatch.to(device))
            if task_key <= 10:
                [pi, sigma, mu] = out_
                pred = ncp.sample(pi, sigma, mu).tolist()[0][0]
                
                tiktok = 0
                while pred < 0 or pred > 1:
                    print('!!!!', tot_time, i, pred)
                    if tiktok > 10:
                        pred = 0
                        break
                    tiktok += 1
                    pred = ncp.sample(pi, sigma, mu).tolist()[0][0]
                samples.append(pred)
                outputs.append(pred)
            #elif task_key <= 10: # non-onehot discrete values
            #    pred_num = ncp.discrete_sample(out_)/255.0
                #print(pred)
            #    samples.append(pred_num)
            #    outputs.append(pred_num)
            elif task_key < 20: # onehot discrete values
                pred_num = ncp.discrete_sample(out_)
                pred = [0] * discrete_dim[task_key]
                pred[pred_num] = 1
                #print(pred)
                samples += pred
                outputs.append(pred_num)
                #if discrete_dim[task_key] == 3 and pred_num > 0:
                #    input()
            else: # special port
                pred_num = ncp.discrete_sample(out_)
                print(task_key, pred_num)
                #print(samples)
                #print(outputs)
                #print(rely_attr[task_key])
                interv = (pred_num-1024) * 100 + 1024
                decode_port = pred_num if pred_num < 1024 else np.random.randint(interv, interv+100)
                print(decode_port/65535.0)
                if decode_port > 65535:
                    print(pred_num, interv)
                    decode_port = 65535
                samples[rely_attr[task_key]-1] = decode_port/65535.0
                outputs[rely_attr[task_key]-1] = decode_port/65535.0
        
        outputs[0] = int(np.ceil(outputs[0] * teDeltamax))
        outputs[1] = int(np.exp(outputs[1] * bytmax))
        outputs[2] = int(np.exp(outputs[2] * pktmax))
        outputs[3] = outputs[3] * tdmax
        for i in [4,5]:
            outputs[i] = int(np.rint(outputs[i] * 65535))
        for i in range(6, 6+4):
            outputs[i] = int(np.rint(outputs[i] * 255))
        tot_time += outputs[0]
        samples = [int(tot_time/3600)/24.0] + samples
        outputs = [int(tot_time/3600)] + outputs
        
        print("samples===>", samples)
        sort_output = []
        sort_output = outputs[0:5]
        if outputs[11] == 0:
            o_sa = this_ip
            o_sp = outputs[5]
            o_da = '.'.join([str(da_i) for da_i in outputs[7:7+4]])
            o_dp = outputs[6]
        else:
            o_sa = '.'.join([str(da_i) for da_i in outputs[7:7+4]])
            o_sp = outputs[6]
            o_da = this_ip
            o_dp = outputs[5]
        sort_output.append(o_sa)
        sort_output.append(o_sp)
        sort_output.append(o_da)
        sort_output.append(o_dp)
        prt = ['TCP', 'UDP', 'Other']
        sort_output.append(prt[outputs[12]])
        if pr_enhance and len(generated_rows) > 0:
            #'byt-1', 'pkt-1', 'pr-1', 'td-1']
            dt_pr = enhance_pr(clf, generated_rows[-1], sort_output)    
            sort_output[-1] = prt[dt_pr]
            samples[-1] = 0
            samples[-2] = 0
            samples[-3] = 0
            samples[-3+dt_pr] = 1
        
        row_num += 1
        print('outputs===>', row_num, tot_time, '==>', outputs)
        print('sort_outputs===>', row_num, tot_time, '==>', sort_output)
        generated_rows.append(sort_output)
        
        for_next_input = torch.FloatTensor([samples])
        #print('for next input tensor', for_next_input)
        minibatch = torch.cat((minibatch[0],for_next_input), 0)
        # print('new_margin', marginal.size())
        minibatch = minibatch[1:]

        #print('after_margin', marginal)
        #input()

    return generated_rows

def gen_flush(generated_rows, idx='0', gen_round=0):
    file_name = gen_file%(idx,gen_round)
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        #writer.writerows([['te', 'teDelta', 'byt','in','out','tcp','udp','other']])
        writer.writerows([['te', 'teDelta', 'byt','pkt','td','sa','sp','da', 'dp', 'pr']])
        writer.writerows(generated_rows)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    if "-drawtest" in sys.argv:
        test_from_csv = CustomDatasetFromCSV(test_file, memory_height+1, traffic_feature_num)
        vali_loader = torch.utils.data.DataLoader(dataset=test_from_csv, batch_size=train_batch_size*2, shuffle=True, \
                      num_workers=16, pin_memory=True)
        for load_idx in range(1, default_load_epoch+1):
            load_model(model, load_idx)
            test_validation(load_idx, vali_loader)
    
    if "-load" in sys.argv:
        #load_model(model, default_load_epoch)
        #load_model(model2, 592)
        #load_model(model, 591)
        pass
    check_cuda()

    if "-train" in sys.argv:
        train()
    
    if "-genfolder" in sys.argv:
        for key in sorted(model_list.keys()):
            l_ep = 500 if key <= 10 else 71
            load_model(key, l_ep, model_list[key])
            model_list[key].eval()
        ten_users = ['42.219.145.151', '42.219.152.127', '42.219.152.238', '42.219.152.246', '42.219.153.113', '42.219.153.115', '42.219.153.140', '42.219.153.146', '42.219.153.154', '42.219.153.158']
        for f in glob.glob('input_data/train_set/*.csv'): 
            this_ip = f.split("_")[-1][:-4]
            #if this_ip not in ten_users:
            #    continue
            print('making test for', f)
            #print(this_ip)
            #input()
            gen_round = 1
            for i in range(gen_round):
                generated_rows = test(this_ip)
                if generated_rows is None:
                    continue
                gen_flush(generated_rows, this_ip, i)

    if "-testone" in sys.argv:
        this_ip = '42.219.145.151'
        generated_rows = test(this_ip)
        if "-gen" in sys.argv:
            gen_flush(generated_rows)

    if "-likelihood" in sys.argv:
        test_from_csv = CustomDatasetFromCSV(test_file, memory_height+1, traffic_feature_num)
        vali_loader = torch.utils.data.DataLoader(dataset=test_from_csv, batch_size=train_batch_size*2, shuffle=True, \
                                      num_workers=16, pin_memory=True)
        for load_idx in range(default_load_epoch, default_load_epoch+1):
            load_model(model, load_idx)
            test_validation(load_idx, vali_loader)
        
        #bins_name_list = list(map(float, config['DEFAULT']['bins'].split(',')))
        ## bins_name_list = [i/bytmax for i in bins_name_list]
        #bin_width = bins_name_list[5] - bins_name_list[4]
        #print('bin_width transfer:', bin_width, bin_width/20.12915933105231)
        #time1 = time.time()
        #calc_likelihood(bin_width/20.12915933105231)
        #time2 = time.time()
        #print(time2-time1)

