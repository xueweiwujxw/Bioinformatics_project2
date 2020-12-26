#!/opt/anaconda3/bin/python3 -Bu
# coding: utf-8

import sys
import time
import json
import pickle
import numpy as np
import torch as pt

from torch import nn, optim
from torch.autograd import Variable
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

from model import *

import os


seqid = sys.argv[1]
devid = int(sys.argv[2])
c_time = sys.argv[3]
batchsize = 12

pt.cuda.set_device(devid)
print("#start!!! ", time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
print('#cuda devices:', devid)

print('#loading data ...')
with open('data_test/train%s.sav' % seqid, 'rb') as f: train = pickle.load(f)
with open('data_test/valid%s.sav' % seqid, 'rb') as f: valid = pickle.load(f)
with open('data_test/ontology00.sav', 'rb') as f: ontology = pickle.load(f)
modelfn = 'output/model-data%s-%s_of_ResNet.pth' % (seqid, c_time)

print('#building model ...')
print('#model: ResNet(depth=2, width=64)')

# model ResNet
class ResBlock(nn.Module):
    def __init__(self, depth, numio):
        super(ResBlock, self).__init__()
        
        resblk = [nn.Sequential(nn.Conv2d(numio, numio, kernel_size=3, padding=1),
                        nn.BatchNorm2d(numio), nn.ReLU(),
                        nn.Conv2d(numio, numio, kernel_size=3, padding=1),
                        nn.BatchNorm2d(numio))
                  for i in range(depth)]
        self.resblk = nn.ModuleList(resblk)

        actblk = [nn.ReLU() for i in range(depth)]
        self.actblk = nn.ModuleList(actblk)

    def forward(self, x):
        for res, act in zip(self.resblk, self.actblk):
            x = act(x + res(x))
        return x
class ResNet(nn.Module):
    def __init__(self, depth, width):
        super(ResNet, self).__init__()
        ndense = width*4

        self.embed = nn.Sequential(nn.Linear(volume*6, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                          nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())
        numin, numout = 1, width
        cnn0 = nn.Sequential(nn.Conv2d(numin, numout, kernel_size=5, padding=2, stride=1),
                      nn.BatchNorm2d(numout), nn.ReLU(),
                      ResBlock(depth*1, numout))
        numin, numout = numout, numout*2
        cnn1 = nn.Sequential(nn.Conv2d(numin, numout, kernel_size=3, padding=1, stride=2),
                      nn.BatchNorm2d(numout), nn.ReLU(),
                      ResBlock(depth*2, numout))
        numin, numout = numout, numout*2
        cnn2 = nn.Sequential(nn.Conv2d(numin, numout, kernel_size=3, padding=1, stride=2),
                      nn.BatchNorm2d(numout), nn.ReLU(),
                      ResBlock(depth*3, numout))
        self.net = nn.Sequential(cnn0, cnn1, cnn2, nn.AdaptiveAvgPool2d((1, 1)))
        self.flatten = nn.Flatten()
        self.out0 = nn.Linear(numout,size0)
        self.out1 = nn.Linear(numout,size1)
        self.out2 = nn.Linear(numout,size2)

    def forward(self, x, mask):
        x1 = x / 3.8; x2 = x1 * x1; x3 = x2 * x1
        xx = pt.cat([x1, x2, x3, 4/3/(1/3+x1), 4/3/(1/3+x2), 4/3/(1/3+x3)], dim=-1)
        embed = self.embed(xx)
        embed = embed.masked_fill_(mask.unsqueeze(2), 0)
        mem = embed.unsqueeze(1)
        mem = self.net(mem)
        mem = self.flatten(mem)
        return self.out0(mem), self.out1(mem), self.out2(mem), mem
# end model

model = ResNet(depth=2, width=64).cuda()
trainloader = iterTrain(train, batchsize, 97)
validloader = iterTest(valid, batchsize)

print('#training model ...')
if seqid == '40':
    numbatch = 1000
elif seqid == '95':
    numbatch = 2000
elif seqid == '00':
    numbatch = 4000
# lr_init, lr_min, numepoch0, numepoch1 = 1e-3, 1e-5, 64, 64*12
lr_init, lr_min, numepoch0, numepoch1 = 1e-3, 1e-5, 64, 64*5
opt = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
sched0 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=lr_min)
sched1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=numepoch0, T_mult=1, eta_min=lr_min)

best_acc = 0
for epoch in range(numepoch1-1):
    t0 = time.perf_counter()

    model.train()
    batch_loss = []
    for i, (x, m, y) in enumerate(trainloader):
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
        idx2 = combinations(range(x.size(0)), 2)
        yaln = pt.cat([y[i, :3] == y[j, :3] for i, j in idx2]).reshape([-1, 3]).float().cuda()
        yy0, yy1, yy2, yycode = model(x, m)

        loss = nn.functional.cross_entropy(yy0, y0) / 3 \
             + nn.functional.cross_entropy(yy1, y1) / 3 \
             + nn.functional.cross_entropy(yy2, y2) / 3 \
             + pt.sqrt(pt.mean(pt.square(yycode))) * 0.1
        opt.zero_grad()
        loss.backward()
        batch_loss.append(float(loss))

        opt.step()
        if epoch+1 < numepoch0: sched0.step(epoch + i/numbatch)
        else: sched1.step(epoch+1 + i/numbatch)
        if i+1 >= numbatch: break

    model.eval()
    batch_acc = []
    for i, (x, m, y) in enumerate(validloader):
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
        yy0, yy1, yy2, _ = model(x, m)
        yy0, yy1, yy2 = pt.argmax(yy0, dim=1), pt.argmax(yy1, dim=1), pt.argmax(yy2, dim=1)

        acc0, acc1, acc2 = float(pt.sum(yy0 == y0)), float(pt.sum(yy1 == y1)), float(pt.sum(yy2 == y2))
        acc01 = float(pt.sum((yy0 == y0) & (yy1 == y1)))
        acc012 = float(pt.sum((yy0 == y0) & (yy1 == y1) & (yy2 == y2)))
        batch_acc.append([acc0, acc1, acc2, acc01, acc012])

        if i+1 >= (len(valid) + batchsize - 1) // batchsize: break

    summary = []
    summary.append(opt.param_groups[0]['lr'])
    summary.append(np.mean(batch_loss))
    summary.extend(np.sum(batch_acc, axis=0) / len(valid) * 100)
    summary.append(time.perf_counter() - t0)
    if summary[-2] > best_acc >= 0:
        print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs\t*' % (epoch+1, *summary))
        pt.save(model.state_dict(), modelfn)
        best_acc = summary[-2]
    else:
        print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs' % (epoch+1, *summary))
# print('#done!!!')
print("#done!!! ", time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))

