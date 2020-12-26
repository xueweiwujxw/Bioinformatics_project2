#!/opt/anaconda3/bin/python3 -Bu
# coding: utf-8

import sys
import random
import json
import pickle
import numpy as np

from glob import glob
from os.path import basename
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

from model import *


seqid = sys.argv[1]
countmin = int(sys.argv[2])
hbondmax = float(sys.argv[3])


def calDist2(model):
    coord = np.zeros((len(model), 3), dtype=np.float32)
    for i in model.keys():
        ii = int(i) - 1
        coord[ii, 0] = model[i]['x']
        coord[ii, 1] = model[i]['y']
        coord[ii, 2] = model[i]['z']
    return squareform(pdist(np.array(coord, dtype=np.float32)))

def isFrag(model, dist2, idx, radius=radius, cutoff=4):
    if model.get(str(idx+1)) is None: return False
    for i in range(radius):
        ii = idx-i-1
        if model.get(str(ii+1)) is None: return False
        if alphabet_res.find(model[str(ii+1)]['res']) < 0: return False
        if dist2[ii, ii+1] > cutoff: return False

        jj = idx+i+1
        if model.get(str(jj+1)) is None: return False
        if alphabet_res.find(model[str(jj+1)]['res']) < 0: return False
        if dist2[jj, jj-1] > cutoff: return False
    return True

def isFrag2(dist2, frag, idx0, idx1, radius=radius, cutoff=8):
    if abs(idx0 - idx1) <= radius: return False
    if not frag[idx0]: return False
    if not frag[idx1]: return False
    if dist2[idx0, idx1] > cutoff: return False
    return True

def buildLabel(remark):
    label = [lab2idx['class'][remark['class']], lab2idx['fold'][remark['fold']],
             lab2idx['super'][remark['super']], lab2idx['family'][remark['family']]]
    label = np.array(label, dtype=np.int32)
    ontology01[label[0], label[1]] = ontology12[label[1], label[2]] = ontology23[label[2], label[3]] = True
    return label

def buildContact(model, idx0, idx1, radius=radius):
    res, ss, acc, dihedral, coord = [], [], [], [], []
    for i in list(range(idx0-radius, idx0+radius+1)) + list(range(idx1-radius, idx1+radius+1)):
        ii = str(i+1)
        res.append(alphabet_res.find(model[ii]['res']))
        ss.append(alphabet_ss.find(model[ii]['ss']))
        acc.append(model[ii]['acc'])
        dihedral.append([model[ii]['phi'], model[ii]['psi']])
        coord.append([model[ii]['x'], model[ii]['y'], model[ii]['z']])
    pos = np.array([idx0, idx1], dtype=np.int32)
    res = np.array(res, dtype=np.int32)
    ss = np.array(ss, dtype=np.int32)
    acc = np.array(acc, dtype=np.float32)
    dihedral = np.array(dihedral, dtype=np.float32)
    coord = np.array(coord, dtype=np.float32)
    coord = coord - np.mean(coord, axis=0)
    return dict(pos=pos, res=res, ss=ss, acc=acc, dihedal=dihedral, coord=coord)


print('#loading SCOP%s data ...' % seqid)
scop = {}
for fn in (glob('scope-2.07-%s/*/*.json' % seqid)):
    fid = basename(fn)[:7]
    with open(fn, 'r') as f:
        scop[fid] = json.load(f)
with open('scope-2.07-00/lab2idx.json', 'r') as f:
    lab2idx = json.load(f)
ontology01 = np.zeros([size0, size1], dtype=np.bool)
ontology12 = np.zeros([size1, size2], dtype=np.bool)
ontology23 = np.zeros([size2, size3], dtype=np.bool)
print('#size:', len(scop))

print('#building contactlib data ...')
data = []
for pdbid in sorted(scop.keys(), key=lambda k: len(scop[k]['model'])):
    model = scop[pdbid]['model']
    size = len(model)
    if size <= 50: continue
    if size >= 1000: continue

    dist2 = calDist2(model)
    frag = np.array([isFrag(model, dist2, i) for i in range(size)], dtype=np.bool)
    frag2 = np.zeros([size, size], dtype=np.bool)
    for idx0, res0 in model.items():
        idx0 = int(idx0)-1

        if float(res0['nho0e']) <= hbondmax:
            idx1 = idx0 + int(res0['nho0p'])
            if isFrag2(dist2, frag, idx0, idx1):
                frag2[idx0, idx1] = True

        if float(res0['nho1e']) <= hbondmax:
            idx2 = idx0 + int(res0['nho1p'])
            if isFrag2(dist2, frag, idx0, idx2):
                frag2[idx0, idx2] = True
    if np.sum(frag2) <= 20: continue

    contact = [buildContact(model, i, j) for i, j in zip(*np.where(frag2))]
    label = buildLabel(scop[pdbid]['remark'])
    release = float(scop[pdbid]['remark']['release'])
    data.append(dict(contact=contact, label=label, pdbid=pdbid, release=release))
print('#size:', len(data))
with open('data_test/data%s.sav' % seqid, 'wb') as f:
    pickle.dump(data, f)
print('#ontology:', np.sum(ontology01), np.sum(ontology12), np.sum(ontology23))
with open('data_test/ontology%s.sav' % seqid, 'wb') as f:
    pickle.dump(dict(ontology01=ontology01, ontology12=ontology12, ontology23=ontology23), f)

print('#splitting train-valid-test data ...')
train, valid, test = [], [], []
random.shuffle(data)
count = {}
for d in data:
    l = d['label'][2]
    if d['release'] < 2.07:
        if count.get(l, 0) < countmin: train.append(d)
        else: valid.append(d)
        count[l] = count.get(l, 0) + 1
    else:
        test.append(d)
trainext, valid = train_test_split(valid, test_size=0.1)
train.extend(trainext)
random.shuffle(train)
test = [d for d in test if count.get(d['label'][2], 0) >= countmin]
print('#size:', len(train), len(valid), len(test))
with open('data_test/train%s.sav' % seqid, 'wb') as f:
    pickle.dump(train, f)
with open('data_test/valid%s.sav' % seqid, 'wb') as f:
    pickle.dump(valid, f)
with open('data_test/test%s.sav' % seqid, 'wb') as f:
    pickle.dump(test, f)

print('#done!!!')

