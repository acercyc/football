import os
os.chdir('/home/acercyc/Project_Araya/football/')
dispPort = !echo $DISPLAY
os.environ['DISPLAY'] = 'localhost:10.0'

import numpy as np

import matplotlib.pyplot as plt

from math import *

from PIL import Image
import glob

import pickle

import lasagne
from lasagne.layers import helper
import theano
import theano.tensor as T

# from permutationlayer import PermutationalLayer

import pandas as pd
import glob





def loadGame(fname):
    players = []
    data = pd.read_csv(fname, sep=',')
    for i in range(11):
        players.append(
            [np.array(data.iloc[:, 3 + i]), np.array(data.iloc[:, 25 + i]), 100 * np.array(data.iloc[:, 47 + i]),
             1000 * np.ones(data.shape[0]), np.zeros(data.shape[0]), np.array(data.iloc[:, 69]),
             np.array(data.iloc[:, 70]), 10 * np.array(data.iloc[:, 71])])
    for i in range(11, 22):
        players.append(
            [np.array(data.iloc[:, 3 + i]), np.array(data.iloc[:, 25 + i]), 100 * np.array(data.iloc[:, 47 + i]),
             np.zeros(data.shape[0]), 1000 * np.ones(data.shape[0]), np.array(data.iloc[:, 69]),
             np.array(data.iloc[:, 70]), 10 * np.array(data.iloc[:, 71])])

    t = np.array(data.iloc[:, 0])
    d = np.array(players).transpose(2, 1, 0) / 1000.0

    return d, t


gn = 0
train_data = []
train_times = []
test_data = []
test_times = []
files = glob.glob('valid_data/*.csv')
files = files[0:3]


for file in files:
    if gn < 130:
        x, y = loadGame(file)
        train_data.append(x)
        train_times.append(y)
    else:
        x, y = loadGame(file)
        test_data.append(x)
        test_times.append(y)
    gn += 1


SITES = 22
VARS = 8
FRAMES = 6
HIDDEN = 192
elu = lasagne.nonlinearities.elu

w = np.array([1, 1, 0, 0, 0, 0, 0, 0])
w = w.reshape((1, VARS, 1))
# w = np.tile(w,SITES)
weights = theano.shared(w.astype(np.float32))

invar = T.tensor3()
targ = T.tensor3()

input = lasagne.layers.InputLayer((None, VARS * FRAMES, SITES), input_var=invar)
l_slice1 = lasagne.layers.SliceLayer(input, axis=1, indices=slice(VARS * (FRAMES - 1), VARS * FRAMES))
l_slice = lasagne.layers.ConcatLayer([l_slice1, l_slice1, l_slice1, l_slice1], axis=1)

# Define subnetwork for 1st layer
dinp_1 = lasagne.layers.InputLayer((None, 2 * VARS * FRAMES, SITES, SITES))
dense1_1 = lasagne.layers.NINLayer(dinp_1, num_units=HIDDEN, nonlinearity=elu)
dense2_1 = lasagne.layers.NINLayer(dense1_1, num_units=HIDDEN, nonlinearity=elu)
dense3_1 = lasagne.layers.NINLayer(dense2_1, num_units=HIDDEN, nonlinearity=elu)
dense4_1 = lasagne.layers.NINLayer(dense3_1, num_units=HIDDEN, nonlinearity=elu)

# Define subnetwork for 2nd layer
dinp2 = lasagne.layers.InputLayer((None, 2 * HIDDEN, SITES, SITES))
dense1_2 = lasagne.layers.NINLayer(dinp2, num_units=HIDDEN, nonlinearity=elu)
dense2_2 = lasagne.layers.NINLayer(dense1_2, num_units=HIDDEN, nonlinearity=elu)
dense3_2 = lasagne.layers.NINLayer(dense2_2, num_units=HIDDEN, nonlinearity=elu)
dense4_2 = lasagne.layers.NINLayer(dense3_2, num_units=HIDDEN, nonlinearity=elu)

# Define subnetwork for 3rd layer
dinp3 = lasagne.layers.InputLayer((None, 2 * HIDDEN, SITES, SITES))
dense1_3 = lasagne.layers.NINLayer(dinp3, num_units=HIDDEN, nonlinearity=elu)
dense2_3 = lasagne.layers.NINLayer(dense1_3, num_units=HIDDEN, nonlinearity=elu)
dense3_3 = lasagne.layers.NINLayer(dense2_3, num_units=HIDDEN, nonlinearity=elu)
dense4_3 = lasagne.layers.NINLayer(dense3_3, num_units=VARS * 4, nonlinearity=None)

perm1 = PermutationalLayer(input, subnet=dense4_1, pooling='max')
perm2 = PermutationalLayer(perm1, subnet=dense4_2)
perm3 = PermutationalLayer(perm2, subnet=dense4_3)
output = lasagne.layers.ElemwiseSumLayer([l_slice, perm3])

out = lasagne.layers.get_output(output)
loss1 = T.mean((out[:, 0:2, :] - targ[:, 0:2, :]) ** 2)
loss2 = T.mean((out[:, 8:10, :] - targ[:, 8:10, :]) ** 2)
loss3 = T.mean((out[:, 16:18, :] - targ[:, 16:18, :]) ** 2)
loss4 = T.mean((out[:, 24:26, :] - targ[:, 24:26, :]) ** 2)

lr = theano.shared(np.cast['float32'](1e-3))
params = lasagne.layers.get_all_params(output, trainable=True)
updates = lasagne.updates.adam(loss1 + loss2 + loss3 + loss4, params, learning_rate=lr)

train = theano.function([invar, targ], [loss1, loss2, loss3, loss4], updates=updates, allow_input_downcast=True)
test = theano.function([invar, targ], [loss1, loss2, loss3, loss4], allow_input_downcast=True)
predict = theano.function([invar], out, allow_input_downcast=True)


# Train network

def formatInputSeq(seq, t):
    inp = np.concatenate([seq[0:-500], seq[100:-400], seq[150:-350], seq[190:-310], seq[199:-301], seq[200:-300]],
                         axis=1)
    trg = np.concatenate([seq[250:-250], seq[300:-200], seq[400:-100], seq[500:]], axis=1)
    #	trg2 = seq[500:]

    idx = (t[500:] - t[:-500] < 600)
    inp = inp[idx]
    trg = trg[idx]

    return inp, trg


for epoch in range(2000):
    lr.set_value(np.cast['float32'](5e-4 * exp(-epoch / 100.0)))
    BS = 1000
    idx = np.random.randint(len(train_data))
    tr_err = np.array([0.0, 0.0, 0.0, 0.0])
    count = 0
    tr_inp, tr_targ = formatInputSeq(train_data[idx], train_times[idx])

    for i in range(tr_inp.shape[0] / BS):
        tr_err += np.array(train(tr_inp[BS * i:BS * (i + 1)], tr_targ[BS * i:BS * (i + 1)]))
        count += 1
    tr_err /= float(count)

    ts_err = np.array([0.0, 0.0, 0.0, 0.0])
    count = 0
    for idx in range(len(test_data)):
        ts_inp, ts_targ = formatInputSeq(test_data[idx], test_times[idx])
        for i in range(ts_inp.shape[0] / BS):
            ts_err += np.array(test(ts_inp[BS * i:BS * (i + 1)], ts_targ[BS * i:BS * (i + 1)]))
            count += 1

    ts_err /= float(count)

    f = open("error.txt", "a")
    f.write("%d %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n" % (
    epoch, tr_err[0], tr_err[1], tr_err[2], tr_err[3], ts_err[0], ts_err[1], ts_err[2], ts_err[3]))
    f.close()

    f = open("prediction1.txt", "wb")
    idx = 0
    ts_inp, ts_targ = formatInputSeq(test_data[idx], test_times[idx])
    for i in range(ts_inp.shape[0] / BS):
        result = predict(ts_inp[BS * i:BS * (i + 1)])
        for j in range(result.shape[0]):
            for k in range(2):
                for l in range(result.shape[2]):
                    f.write("%.6g %.6g " % (result[j, k, l], ts_targ[BS * i + j, k, l]))
            f.write("\n")
    f.close()

    f = open("prediction2.txt", "wb")
    idx = 0
    ts_inp, ts_targ = formatInputSeq(test_data[idx], test_times[idx])
    for i in range(ts_inp.shape[0] / BS):
        result = predict(ts_inp[BS * i:BS * (i + 1)])
        for j in range(result.shape[0]):
            for k in range(2):
                for l in range(result.shape[2]):
                    f.write("%.6g %.6g " % (result[j, k + 8, l], ts_targ[BS * i + j, k + 8, l]))
            f.write("\n")
    f.close()

    f = open("prediction3.txt", "wb")
    idx = 0
    ts_inp, ts_targ = formatInputSeq(test_data[idx], test_times[idx])
    for i in range(ts_inp.shape[0] / BS):
        result = predict(ts_inp[BS * i:BS * (i + 1)])
        for j in range(result.shape[0]):
            for k in range(2):
                for l in range(result.shape[2]):
                    f.write("%.6g %.6g " % (result[j, k + 16, l], ts_targ[BS * i + j, k + 16, l]))
            f.write("\n")
    f.close()

    f = open("prediction4.txt", "wb")
    idx = 0
    ts_inp, ts_targ = formatInputSeq(test_data[idx], test_times[idx])
    for i in range(ts_inp.shape[0] / BS):
        result = predict(ts_inp[BS * i:BS * (i + 1)])
        for j in range(result.shape[0]):
            for k in range(2):
                for l in range(result.shape[2]):
                    f.write("%.6g %.6g " % (result[j, k + 24, l], ts_targ[BS * i + j, k + 24, l]))
            f.write("\n")
    f.close()

    pickle.dump(lasagne.layers.get_all_param_values(output), open("network.params", "wb"))
