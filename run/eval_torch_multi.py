#!/usr/bin/env python
import numpy as np
import argparse
import sys, os
import subprocess
import csv, yaml
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
torch.set_num_threads(nthreads)

parser = argparse.ArgumentParser()
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--lumi', action='store', type=float, default=138, help='Reference luminosity in fb-1')
parser.add_argument('-d', '--input', action='store', type=str, required=True, help='directory with pretrained model parameters')
parser.add_argument('--model', action='store', choices=('none', 'default', 'log3ch', 'log5ch', 'original', 'circpad', 'circpadlog3ch', 'circpadlog5ch'),
                               default='none', help='choice of model')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('-c', '--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')

args = parser.parse_args()
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
lumiVal = args.lumi

predFile = args.input+'/prediction.csv'
import pandas as pd

sys.path.append("../python")

print("Load data", end='')
from HEPCNN.dataset_hepcnn import HEPCNNDataset as MyDataset

myDataset = MyDataset()
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    myDataset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    myDataset.setProcessLabel(name, sampleInfo['label'])
myDataset.initialize()
print("done")

print("Split data", end='')
lengths = [int(0.6*len(myDataset)), int(0.2*len(myDataset))]
lengths.append(len(myDataset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDataset, valDataset, testDataset = torch.utils.data.random_split(myDataset, lengths)
torch.manual_seed(torch.initial_seed())
print("done")

kwargs = {'num_workers':min(config['training']['nDataLoaders'], nthreads)}
if args.device >= 0:
    torch.cuda.set_device(args.device)
    if torch.cuda.is_available():
        #if hvd: kwargs['num_workers'] = 1
        kwargs['pin_memory'] = True

testLoader = DataLoader(testDataset, batch_size=args.batch, shuffle=False, **kwargs)

print("Load model", end='')
if args.model == 'none':
    print("Load saved model from", (args.input+'/model.pth'))
    model = torch.load(args.input+'/model.pth', map_location='cpu')
else:
    print("Load the model", args.model)
    if args.model == 'original':
        from HEPCNN.torch_model_original import MyModel
    elif 'circpad' in args.model:
        from HEPCNN.torch_model_circpad import MyModel
    else:
        from HEPCNN.torch_model_default import MyModel
    model = MyModel(testDataset.width, testDataset.height, model=args.model)

device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
print('done')

model.load_state_dict(torch.load(args.input+'/weight_0.pth', map_location='cpu'))
model.to(device)
print('modify model', end='')
model.fc.add_module('output', torch.nn.Sigmoid())
model.eval()
print('done')

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

labels, preds = [], []
weights, scaledWeights = [], []
true_label = torch.Tensor()
pred_label = torch.Tensor()
print(true_label)
print(true_label.shape)
print(pred_label)
confusion_matrix_ = torch.zeros(6,6)
for i, (data, label, weight, rescale, _) in enumerate(tqdm(testLoader)):
    data = data.float().to(device)
    weight = weight.float()
    pred = model(data).detach().to('cpu')

    value, index = torch.max(pred, dim = 1)
    true_label = torch.cat([true_label, label], dim=0)
    pred_label = torch.cat([pred_label, index], dim=0)
    # confusion_matrix_ = confusion_matrix_ + confusion_matrix(label,index,labels=[0,1,2,3,4,5])
confusion_matrix_ = confusion_matrix(true_label,pred_label,normalize="true")
print(confusion_matrix_)

axis_list = [r"$\tau^{\mp}\rightarrow\pi^{\mp}\nu$",r"$\tau^{\mp}\rightarrow\pi^{\mp}\pi^{\pm}\pi^{\mp}\nu$",r"$\tau^{\mp}\rightarrow\pi^{\mp}\pi^{\pm}\pi^{\mp}\pi^{0}\nu$",r"$\tau^{\mp}\rightarrow\pi^{\mp}\pi^{0}\nu$",r"$\tau^{\mp}\rightarrow\pi^{\mp}\pi^{0}\pi^{0}\nu$",r"$Z\rightarrow q \bar{q}$"]

df_cm = pd.DataFrame(confusion_matrix_, index = [i for i in axis_list], columns = [i for i in axis_list])
plt.figure(figsize = (11,10))
ax = sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
plt.savefig("conf.png")
