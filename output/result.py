import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

path = "./"

files = os.listdir(path)
files = list(filter(lambda x:x[-4:] == '.log', files))
testfiles = list(filter(lambda x:x[-9:] == '_test.log', files))
out4file = list(filter(lambda x:x[-9:] == '_out4.log', files))
files = list(filter(lambda x:x[-9:] != '_test.log', files))
files = list(filter(lambda x:x[-9:] != '_out4.log', files))
files = list(filter(lambda x:x[10:12] != '00', files))
print(testfiles)

results = {}
tresults = {}
o4results = {}

def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'

for item in files:
    maxacc = 0.0
    maxaccs = []
    f = open(item)
    name = item[27:-4]
    loss, acc0, acc1, acc2, acc3, acc4 = [], [], [], [], [], []
    while True:
        line = f.readline()
        if not line:
            break
        lists = line.split()
        if lists[0].find('epoch') != -1:
            loss.append(float(lists[2]))
            acc0.append(float(lists[3][:-1])/100)
            acc1.append(float(lists[4][:-1])/100)
            acc2.append(float(lists[5][:-1])/100)
            acc3.append(float(lists[6][:-1])/100)
            acc4.append(float(lists[7][:-1])/100)
            if maxacc < float(lists[7][:-1])/100:
                maxaccs = ((round(float(lists[3][:-1])/100, 4),
                            round(float(lists[4][:-1])/100, 4),
                            round(float(lists[5][:-1])/100, 4),
                            round(float(lists[6][:-1])/100, 4),
                            round(float(lists[7][:-1])/100, 4),))
            maxacc = max(float(lists[7][:-1])/100, maxacc)
    res = zip(loss, acc0, acc1, acc2, acc3, acc4)
    res = list(res)
    res = np.array(res)
    results.setdefault(name, res)
    print(name, maxaccs)
for name, res in results.items():
    accs = ['acc0', 'acc1', 'acc2', 'acc01', 'acc012']
    plt.ylim(bottom=0, top=1)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(0, 320, 25))
    plt.title("Accuracy of "+name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    for ii in range(1, 6):
        plt.plot(np.arange(1, 320, step=1), res[:, ii], label=accs[ii-1])
        plt.legend()
    plt.show()

for name, res in results.items():
    plt.ylim(bottom=0, top=6)
    plt.yticks(np.arange(0, 6, 1))
    plt.xticks(np.arange(0, 320, 25))
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.arange(1, 320, step=1), res[:, 0], label=name)
    plt.legend()
plt.show()

for name, res in results.items():
    plt.ylim(bottom=0, top=1)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(0, 320, 25))
    plt.title("Accuracy012")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy012')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.plot(np.arange(1, 320, step=1), res[:, 5], label=name)
    plt.legend()
plt.show()

for item in testfiles:
    maxacc = 0.0
    maxaccs = []
    f = open(item)
    if item.find('data00') != -1:
        name = item[27:-4] + '00'
    elif item.find('data40') != -1:
        name = item[27:-4] + '40'
    loss, acc0, acc1, acc2, acc3, acc4, tacc0, tacc1, tacc2, tacc3, tacc4 = [], [], [], [], [], [], [], [], [], [], []
    while True:
        line = f.readline()
        if not line:
            break
        lists = line.split()
        if lists[0].find('epoch') != -1:
            loss.append(float(lists[2]))
            acc0.append(float(lists[3][:-1])/100)
            acc1.append(float(lists[4][:-1])/100)
            acc2.append(float(lists[5][:-1])/100)
            acc3.append(float(lists[6][:-1])/100)
            acc4.append(float(lists[7][:-1])/100)
            tacc0.append(float(lists[8][:-1]) / 100)
            tacc1.append(float(lists[9][:-1]) / 100)
            tacc2.append(float(lists[10][:-1]) / 100)
            tacc3.append(float(lists[11][:-1]) / 100)
            tacc4.append(float(lists[12][:-1]) / 100)
            if maxacc < float(lists[12][:-1])/100:
                maxaccs = ((round(float(lists[3][:-1])/100, 4),
                            round(float(lists[4][:-1])/100, 4),
                            round(float(lists[5][:-1])/100, 4),
                            round(float(lists[6][:-1])/100, 4),
                            round(float(lists[7][:-1])/100, 4),
                            round(float(lists[8][:-1]) / 100, 4),
                            round(float(lists[9][:-1]) / 100, 4),
                            round(float(lists[10][:-1]) / 100, 4),
                            round(float(lists[11][:-1]) / 100, 4),
                            round(float(lists[12][:-1]) / 100, 4)))
            maxacc = max(float(lists[12][:-1])/100, maxacc)
    res = zip(loss, acc0, acc1, acc2, acc3, acc4, tacc0, tacc1, tacc2, tacc3, tacc4)
    res = list(res)
    res = np.array(res)
    tresults.setdefault(name, res)
    print(name, maxaccs)

for name, res in tresults.items():
    accs = ['acc0', 'acc1', 'acc2', 'acc01', 'acc012', 'tacc0', 'tacc1', 'tacc2', 'tacc3', 'tacc4']
    plt.ylim(bottom=0, top=1)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(0, 320, 25))
    plt.title("Accuracy of "+name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    for ii in range(1, 11):
        plt.plot(np.arange(1, 320, step=1), res[:, ii], label=accs[ii-1])
        plt.legend()
    plt.show()

for item in out4file:
    maxacc = 0.0
    maxaccs = []
    f = open(item)
    name = item[27:-4] + ' predict family'
    loss, acc0, acc1, acc2, acc3, acc4, acc5 = [], [], [], [], [], [], []
    while True:
        line = f.readline()
        if not line:
            break
        lists = line.split()
        if lists[0].find('epoch') != -1:
            loss.append(float(lists[2]))
            acc0.append(float(lists[3][:-1])/100)
            acc1.append(float(lists[4][:-1])/100)
            acc2.append(float(lists[5][:-1])/100)
            acc3.append(float(lists[6][:-1])/100)
            acc4.append(float(lists[7][:-1])/100)
            acc5.append(float(lists[8][:-1])/100)
            if maxacc < float(lists[8][:-1])/100:
                maxaccs = ((round(float(lists[3][:-1])/100, 4),
                            round(float(lists[4][:-1])/100, 4),
                            round(float(lists[5][:-1])/100, 4),
                            round(float(lists[6][:-1])/100, 4),
                            round(float(lists[7][:-1])/100, 4),
                            round(float(lists[8][:-1])/100, 4)))
            maxacc = max(float(lists[8][:-1])/100, maxacc)
    res = zip(loss, acc0, acc1, acc2, acc3, acc4, acc5)
    res = list(res)
    res = np.array(res)
    o4results.setdefault(name, res)
    print(name, maxaccs)

for name, res in o4results.items():
    accs = ['acc0', 'acc1', 'acc2', 'acc01', 'acc012', 'acc0123']
    plt.ylim(bottom=0, top=1)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(0, 320, 25))
    plt.title("Accuracy of "+name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    for ii in range(1, 7):
        plt.plot(np.arange(1, 320, step=1), res[:, ii], label=accs[ii-1])
        plt.legend()
    plt.show()

