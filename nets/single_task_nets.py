import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F
import math
from scipy import integrate
import numpy as np
import os

class SingleTaskModel(object):
    def __init__(self, target_type='continuous'):
        self.net = SingleTaskNet(target_type)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.9)


class SingleTaskNet(nn.Module):
    def __init__(self, target_type='continuous', dim=2):
        super(SingleTaskNet, self).__init__()

        enc_cfgs = {
            'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
            'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
            'D': [64, 64, 'M', 128, 128, 'M']
        }
        mdn_cfgs = {
            'in_num': 128*4, #512, #*4,
            'out_num': 1, #5 + 4,
            'comp_num': 10,
            'bin_width': 0.0027,
        }
        self.target_type = 0 if target_type == 'continuous' else 1
        self.dim = dim
        self.mdn_cfgs = mdn_cfgs

        self.enc = self.make_layers(enc_cfgs['D'])
        # self.mdn =
        if self.target_type == 0:
            self.pi_layer, self.sigma_layer, self.mu_layer = self.make_mdns()  
        else: 
            self.dec = self.make_decs(self.dim, is_onehot=True)

    def forward(self, minibatch):
        input_ = self.enc_forward(minibatch)
        #input_ = torch.unsqueeze(minibatch, 1)
        if self.target_type == 0:
            pi, sigma, mu = self.mdn_forward(input_)
            out_ = [pi, sigma, mu]
        else:
            out_ = self.dec_forward(input_)
        return out_
        
    def enc_forward(self, minibatch):
        #print('debug',minibatch.size())
        x = torch.unsqueeze(minibatch, 1)       # [B, C, T, E] Add a channel dim.
        #print('debug',x.size())
        x = self.enc(x)
        out = x.view(x.size(0), -1)       # [B, F * window]
        #print('conv out', out.size())
        return out

    def mdn_forward(self, input_):
        pi = self.pi_layer(input_)
        sigma = torch.exp(self.sigma_layer(input_)).view(-1, self.mdn_cfgs['comp_num'], 1)
        mu = self.mu_layer(input_).view(-1, self.mdn_cfgs['comp_num'], 1)
        return pi, sigma, mu

    def dec_forward(self, input_):
        #print('11111111111111111111111111')
        out = self.dec(input_)
        return out

    def make_mdns(self):
        pi = nn.Sequential(
            nn.Linear(self.mdn_cfgs['in_num'], self.mdn_cfgs['comp_num']),
            nn.Softmax(dim=1)
        )
        sigma = nn.Sequential(
                nn.Linear(self.mdn_cfgs['in_num'], 1*self.mdn_cfgs['comp_num']),
                #nn.ReLU(inplace=True),
                #nn.Linear(32, 1*self.mdn_cfgs['comp_num']),
                #nn.ReLU(inplace=True)
                )
        mu = nn.Sequential(
                nn.Linear(self.mdn_cfgs['in_num'], 1*self.mdn_cfgs['comp_num']),
                #nn.ReLU(inplace=True),
                #nn.Linear(32, 1*self.mdn_cfgs['comp_num']),
                #nn.ReLU(inplace=True)
                )
        return pi, sigma, mu

    def make_decs(self, output_num, hidden_num=100, is_onehot=True):
        input_num = self.mdn_cfgs['in_num']
        if is_onehot:
            dec_layer = nn.Sequential(
                #nn.Linear(input_num, hidden_num),
                #nn.ReLU(inplace=True),
                #nn.Linear(hidden_num, output_num),
                nn.Linear(input_num, output_num),
                #nn.Softmax()
            )
        else:
            dec_layer = nn.Sequential(
                nn.Linear(input_num, hidden_num),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_num, output_num),
                nn.ReLU(inplace=True)
            )
        return dec_layer

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        print(cfg)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def save_model(self, name, epoch):
        if not os.path.exists('./saved_model'):
            os.makedirs('./saved_model')
        if not os.path.exists('./saved_model/model_%d'%name):
            os.makedirs('./saved_model/model_%d'%name)
        checkpoint = './saved_model/model_%d'%name + '/ep%d.pkl'%epoch
        torch.save(self.state_dict(), checkpoint)
        return self

    def load_model(self, name, epoch):
        checkpoint = './saved_model/model_%d'%name + '/ep%d.pkl'%epoch
        self.load_state_dict(torch.load(checkpoint))
        return self

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

def gaussian_probability(sigma, mu, target, bin_width):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    if bin_width == None:
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    else:
        delta_cdf = bin_width / 2.0
        ret = cdf_func(sigma, mu, target+delta_cdf) - cdf_func(sigma, mu, target-delta_cdf)
    # ret = integrate.quad(pdf_func, target, target+)
    if (torch.isnan(ret).any()):
        ret[ret != ret] = 0
    return torch.prod(ret, 2)

def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # print("sample: pi:", pi.size(), pi)
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample

def discrete_sample(out_):
    #print(out_)
    probs = F.softmax(out_, dim=1)
    dist = torch.distributions.Categorical(probs)
    sample = dist.sample().data.tolist()[0]
    #print(probs)
    #print('sample', sample)
    return sample

#def forward(self, out):
#        one_hot = self.oh(out)
#        return one_hot
# torch.max(out, 1)[1]

def dec_loss(pred, target, is_onehot=False):
    # _, targets = y1.max(dim=0)
    # nn.CrossEntropyLoss()(out, Variable(targets)) 
    #prediction = torch.max(out, 1)[1]
    #pred_y = prediction.data.numpy()
    #target_y = y.data.numpy()
    #print('pred', pred.size(), pred[:5])
    
    #print('target_', target.size(), target[:5])
    #print('target', target.size())
    if is_onehot:
        target = torch.max(target, 1)[1].long()
    #print(target.size())
    #print('target_', target.size(), target[:5])
    loss = torch.nn.CrossEntropyLoss()(pred, target)
    #print(loss)
    #input()
    return loss

def bce_loss(pred, target):
    return torch.nn.BCEWithLogitsLoss()(pred, target)

def mdn_loss(model, pi, sigma, mu, target, bin_width=None):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target, bin_width)
    nll = -torch.log(torch.sum(prob, dim=1)+1e-10)
    return torch.mean(nll)

def pdf_func(sigma, mu, target):
    return ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma 

def cdf_func(sigma, mu, value):
    return 0.5 * (1 + torch.erf((value - mu) * sigma.reciprocal() / math.sqrt(2)))
