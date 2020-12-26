#!/opt/anaconda3/bin/python3 -Bu
# coding: utf-8

import math
import random
import numpy as np
import torch as pt

from torch import nn, optim
from itertools import combinations
from scipy.spatial.distance import pdist, squareform


alphabet_res = 'LAGVESKIDTRPNFQYHMCW'
alphabet_ss = 'HE TSGBI'
radius, diameter, volume = 2, 5, 45
size0, size1, size2, size3 = 12, 1457, 2322, 5258


def iterTrain(data, batchsize, bucketsize, noiserate=0.05):
    while True:
        bucket = sorted(random.sample(data, batchsize*bucketsize), key=lambda k: len(k['contact']))
        bucket = [bucket[i:i+batchsize] for i in range(0, len(bucket), batchsize)]
        random.shuffle(bucket)

        for batch in bucket:
            noise = np.random.normal(0, noiserate, [len(batch), len(batch[-1]['contact']), 10, 3])
            seq = np.zeros([len(batch), len(batch[-1]['contact']), volume], dtype=np.float32)
            mask = np.ones([len(batch), len(batch[-1]['contact'])], dtype=np.bool)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int32)
            for i, b in enumerate(batch):
                size = len(b['contact'])
                seq[i, :size] = np.array([pdist(b['contact'][j]['coord'] + noise[i, j]) for j in range(size)])
                mask[i, :size] = False
                label[i] = b['label']
            seq = pt.from_numpy(seq).float()
            mask = pt.from_numpy(mask).bool()
            label = pt.from_numpy(label).long()
            yield seq, mask, label

def iterTest(data, batchsize):
    while True:
        bucket = sorted(data, key=lambda k: len(k['contact']))
        bucket = [bucket[i:i+batchsize] for i in range(0, len(bucket), batchsize)]

        for batch in bucket:
            seq = np.zeros([len(batch), len(batch[-1]['contact']), volume], dtype=np.float32)
            mask = np.ones([len(batch), len(batch[-1]['contact'])], dtype=np.bool)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int32)
            for i, b in enumerate(batch):
                size = len(b['contact'])
                seq[i, :size] = np.array([pdist(b['contact'][j]['coord']) for j in range(size)])
                mask[i, :size] = False
                label[i] = b['label']
            seq = pt.from_numpy(seq).float()
            mask = pt.from_numpy(mask).bool()
            label = pt.from_numpy(label).long()
            yield seq, mask, label


class LinearBlock(nn.Module):
    def __init__(self, nio, nhid):
        super(LinearBlock, self).__init__()

        self.dense = nn.Sequential(nn.Linear(nio, nhid), nn.LayerNorm(nhid), nn.ReLU(),
                                 nn.Linear(nhid, nio), nn.LayerNorm(nio), nn.ReLU())

    def forward(self, x):
        return self.dense(x)

class LinearNet(nn.Module):
    def __init__(self, depth, width):
        super(LinearNet, self).__init__()
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4

        self.embed = nn.Sequential(nn.Linear(volume*6, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        self.encod = nn.Sequential(*[LinearBlock(width, ndense) for i in range(depth)])

        self.out0 = nn.Linear(width, size0)
        self.out1 = nn.Linear(width, size1)
        self.out2 = nn.Linear(width, size2)

    def forward(self, x, mask):
        x1 = x / 3.8; x2 = x1 * x1; x3 = x2 * x1
        xx = pt.cat([x1, x2, x3, 4/3/(1/3+x1), 4/3/(1/3+x2), 4/3/(1/3+x3)], dim=-1)
        embed = self.embed(xx)

        mem = self.encod(embed)
        mem = mem.masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.unsqueeze(2).float().sum(1))

        return self.out0(mem), self.out1(mem), self.out2(mem), mem

