#added imports and config from train_models.py but will clean up later

import argparse
import torch
from torch import autograd
from tensorboardX import SummaryWriter
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

import numpy as np
import utils_config

from matplotlib import pyplot as plt
import PIL.Image

from scipy.stats import median_abs_deviation

from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

import sys

from utils import *
from datasets import *
from cf_algos import *

import datetime
from copy import deepcopy


config_file_d="./conf/datasets.json"

config_d = utils_config.load_config(config_file_d)
config_d = utils_config.serialize_config(config_d)


parser = argparse.ArgumentParser()

 #add arguments here:
parser.add_argument('--dataset', type=str, default='german', help='dataset name')
parser.add_argument('--cf_algo', type=str, default='wachter', help='counterfactual algorithm name')
parser.add_argument('--model_path', type=str, default='models/model.pth', help='path to the model')
parser.add_argument("--hidden", default=200, type=int, help="Number of hidden units per layer")


 #########
args = parser.parse_args()

dataset = args.dataset
CFNAME = args.cf_algo
HIDDEN = args.hidden
TARGET = 1.0
lmbda = 1.0
torch.manual_seed(10)
np.random.seed(0)
config = {}
config['lmbda'] = lmbda
config['TARGET'] = TARGET
writer = None

#get presplit data also, preprocess it
data, labels, protected, data_t, labels_t, protected_t, cat_features = get_presplit_data(dataset)
config['cat_features'] = cat_features
numerical = np.array([val for val in range(data.shape[1]) if val not in cat_features])
ss = StandardScaler()
data = ss.fit_transform(data)
data_t = ss.transform(data_t)

mads = []
for c in range(data.shape[1]):
	mad_c = median_abs_deviation(data[:,c], scale='normal')
	if mad_c == 0:
		mads.append(1)
	else:
		mads.append(mad_c)

if CFNAME == "wachter" or CFNAME == "dice":
	config['mad'] = torch.from_numpy(np.array(mads)).cuda()
else:
	config['mad'] = None




# Get objective and distance function
df, objective = get_obj_and_df(CFNAME)

## Setup data  ###########################
data = torch.from_numpy(data).float().cuda()
labels = torch.from_numpy(labels).float().cuda()
protected = torch.from_numpy(protected).float().cuda()
data_t = torch.from_numpy(data_t).float().cuda()
labels_t = torch.from_numpy(labels_t).float().cuda()
protected_t = torch.from_numpy(protected_t).float().cuda()
data.requires_grad = True
protected.requires_grad = True
##########################################

## Setup model ##########################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, return_logit=False):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        out1 = self.fc4(out)
        out = self.sigmoid(out1)

        if return_logit:
        	return out1
        else:
        	return out
###############################################
# Load Model
model = NeuralNet(data.shape[1], HIDDEN, 1).cuda()
model.load_state_dict(torch.load(args.model_path + '_model.pth'))
model.eval()

# load noise
noise = torch.load(args.model_path + '_noise.pt')
noise = noise.cuda()
noise.requires_grad = True



#actual eval:
#first testing accuracy again
neg_not_pro = negative_not_protected_indices(data_t, model, protected_t)
final_preds = (model(data_t[neg_not_pro] + noise)[:,0] > 0.5 ).int()
succ = (torch.sum(torch.abs(model(data_t[neg_not_pro] + noise)[:,0] > 0.5)) / torch.sum(neg_not_pro))
print ("Noise norm",torch.norm(noise,p=1))
print ("Delta flip success", succ)
final_preds = (model(data_t)[:,0] > 0.5).int() 
print ("Testing Accuracy", torch.sum(final_preds == labels_t) / final_preds.shape[0])
print ('#######')



assess(
        1, 
        model, 
        data, 
        protected, 
        labels, 
        data_t, 
        protected_t, 
        labels_t, 
        CFNAME, 
        config, 
        writer, 
        noise, 
        False,
        df=df,
        verbose=True,
        r=True)