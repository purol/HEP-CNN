#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self, width, height, **kwargs):
        super(MyModel, self).__init__()
        model = "default" if "model" not in kwargs else kwargs["model"]
        self.fw = width//2//2//2 # if input is 256, this value is 32
        self.fh = height//2//2//2

        self.nch = 5 if '5ch' in model else 4
        self.doLog = ('log' in model)
        if 'norm0' in model: self.doNorm = 0b0 ## do not normalize at all
        elif 'norm1' in model: self.doNorm = 0b1111 ## normalize all, 1111
        else: self.doNorm = 0b101 ## The default normalization: ecal and tracker
        self.doCat = ('cat' in model)

        self.conv_first = nn.Sequential(
            nn.Conv2d(self.nch//2, 64, kernel_size=(7, 7), stride=1, padding=3), # input: c=2 280*280, out: c=64 280*280
            # input: c=3 224*224, out: c=64 224*224 
            nn.MaxPool2d(kernel_size=(2, 2)), # out: c=64 140*140
            # input: out: c=64 112*112
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=1, padding=3), # input: c=64 140*140, out: c=128 141*141
            # input: out: c=64 113*113
            nn.MaxPool2d(kernel_size=(2, 2)), # input: c=128 281*281, out: c=128 70*70
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, kernel_size=(7, 7), stride=1, padding=3), # input: c=128 70*70, out: c=256 71*71
            nn.MaxPool2d(kernel_size=(2, 2)), # input: c=256 71*71, out: c=256 35*35
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            #nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=2),
            nn.Conv2d(256, 256, kernel_size=(7, 7), stride=1, padding=3), # input: c=256 35*35, out: c=256 35*35
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),

        )
        self.conv_second = nn.Sequential(
            nn.Conv2d(self.nch//2, 64, kernel_size=(3, 3), stride=1, padding=1), # input: c=2 280*280, out: c=64 280*280
            # input: c=3 224*224, out: c=64 224*224 
            nn.MaxPool2d(kernel_size=(2, 2)), # out: c=64 140*140
            # input: out: c=64 112*112
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1), # input: c=64 140*140, out: c=128 141*141
            # input: out: c=64 113*113
            nn.MaxPool2d(kernel_size=(2, 2)), # input: c=128 281*281, out: c=128 70*70
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1), # input: c=128 70*70, out: c=256 71*71
            nn.MaxPool2d(kernel_size=(2, 2)), # input: c=256 71*71, out: c=256 35*35
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            #nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1), # input: c=256 35*35, out: c=256 35*35
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fw*self.fh*512 + (3 if self.doCat else 0), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # dropout: remove connection randomly
            nn.Linear(512, 1),
            # Linear(input, output)
            #nn.Sigmoid(),
        )

    def forward(self, x):
        n, c = x.shape[0], x.shape[1]
        if c > 6: ## We don't expect image more than 6 channel, this indicates that the image format was NHWC.
            x = x.permute(0,3,1,2)
            c = x.shape[1]
        s, _ = torch.max(x.reshape(n,c,-1), dim=-1)
        if self.doNorm &   0b1 != 0: x[:,0,:,:] /= 91200
        if self.doNorm &  0b10 != 0: x[:,1,:,:] /= 91200
        if self.doNorm & 0b100 != 0: x[:,2,:,:] /= 91200
        if self.doNorm & 0b1000!= 0: x[:,3,:,:] /= 91200
        # 0b means binary
        if self.nch == 5:
            xx = x[:,:2,:,:]
            x = torch.cat((x, xx), dim=1)
        if self.doLog:
            x[:,:2,:,:] = torch.log10(x[:,:2,:,:]/1e-5+1)

        # NCHW
        x_first = x[:,:2,:,:]
        x_second = x[:,2:,:,:]
       
 
        x_first = self.conv_first(x_first) # c=256 32*32
        x_second = self.conv_second(x_second)
        
        x_cat = torch.cat([x_first, x_second], dim=1) # c=512 32*32
        
        x_cat = x_cat.flatten(start_dim=1)
        # x_first = x_first.flatten(start_dim = 1)
        if self.doCat: x = torch.cat([x, s], dim=-1)
        x_cat = self.fc(x_cat)
        #x_first = self.fc(x_first)
        return x_cat
        #return x_first
