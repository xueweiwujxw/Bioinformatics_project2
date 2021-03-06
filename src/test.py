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

# devids = "1,4,7"
# os.environ["CUDA_VISIBLE_DEVICES"]=devids


seqid = sys.argv[1]
devid = int(sys.argv[2])
c_time = sys.argv[3]
batchsize = 12

pt.cuda.set_device(devid)
# print('#cuda devices:', devid)
print("#start!!! ", time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))
print('#cuda devices:', devid)

print('#loading data ...')
with open('data_test/train%s.sav' % seqid, 'rb') as f: train = pickle.load(f)
with open('data_test/valid%s.sav' % seqid, 'rb') as f: valid = pickle.load(f)
with open('data_test/ontology00.sav', 'rb') as f: ontology = pickle.load(f)
# modelfn = 'output/model-data%s-dev%d.pth' % (seqid, devid)
modelfn = 'output/model-data%s-%s_of_LinearNet.pth' % (seqid, c_time)

print('#building model ...')
print('#model: LinearNet(depth=4, width=512)')
model = LinearNet(depth=4, width=512).cuda()
# model = pt.nn.DataParallel(model, device_ids=[0,1,2])
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
for epoch in range(1):
#     t0 = time.perf_counter()

#     model.train()
#     batch_loss = []
#     for i, (x, m, y) in enumerate(trainloader):
#         x = Variable(x).cuda()
#         m = Variable(m).cuda()
#         y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
#         idx2 = combinations(range(x.size(0)), 2)
#         yaln = pt.cat([y[i, :3] == y[j, :3] for i, j in idx2]).reshape([-1, 3]).float().cuda()
#         yy0, yy1, yy2, yycode = model(x, m)

#         loss = nn.functional.cross_entropy(yy0, y0) / 3 \
#              + nn.functional.cross_entropy(yy1, y1) / 3 \
#              + nn.functional.cross_entropy(yy2, y2) / 3 \
#              + pt.sqrt(pt.mean(pt.square(yycode))) * 0.1
#         opt.zero_grad()
#         loss.backward()
#         batch_loss.append(float(loss))

#         opt.step()
#         if epoch+1 < numepoch0: sched0.step(epoch + i/numbatch)
#         else: sched1.step(epoch+1 + i/numbatch)
#         if i+1 >= numbatch: break

#     model.eval()
#     batch_acc = []
#     for i, (x, m, y) in enumerate(validloader):
#         x = Variable(x).cuda()
#         m = Variable(m).cuda()
#         y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
#         yy0, yy1, yy2, _ = model(x, m)
#         yy0, yy1, yy2 = pt.argmax(yy0, dim=1), pt.argmax(yy1, dim=1), pt.argmax(yy2, dim=1)

#         acc0, acc1, acc2 = float(pt.sum(yy0 == y0)), float(pt.sum(yy1 == y1)), float(pt.sum(yy2 == y2))
#         acc01 = float(pt.sum((yy0 == y0) & (yy1 == y1)))
#         acc012 = float(pt.sum((yy0 == y0) & (yy1 == y1) & (yy2 == y2)))
#         batch_acc.append([acc0, acc1, acc2, acc01, acc012])

#         if i+1 >= (len(valid) + batchsize - 1) // batchsize: break

#     summary = []
#     summary.append(opt.param_groups[0]['lr'])
#     summary.append(np.mean(batch_loss))
#     summary.extend(np.sum(batch_acc, axis=0) / len(valid) * 100)
#     summary.append(time.perf_counter() - t0)
#     if summary[-2] > best_acc >= 0:
#         print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs\t*' % (epoch+1, *summary))
#         pt.save(model.state_dict(), modelfn)
#         best_acc = summary[-2]
#     else:
#         print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs' % (epoch+1, *summary))
    maxlength = 0
    for i, (x, m, y) in enumerate(trainloader):
        print(x.shape)
        maxlength = max(maxlength, x.size(1))
print(maxlength)
# print('#done!!!')
# print("#done!!! ", time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime()))

