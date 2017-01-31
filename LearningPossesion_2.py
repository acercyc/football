import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import football.lib as fb
import football.dataGenerator as gen

import football.shelve_ext as she

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.utils.np_utils import to_categorical


posi_team1, posi_team2, posi_ball = gen.genHaifFieldPosition(10 ** 5)
posi_team1 = posi_team1 / fb.dataInfo.fieldSize_x[1]
posi_team2 = posi_team2 / fb.dataInfo.fieldSize_x[1]
posi_ball = posi_ball / fb.dataInfo.fieldSize_x[1]

y_sim = posi_ball[:, 0] > 0
y_sim = to_categorical(y_sim)


def genModel():
    m = Sequential()
    m.add(Dense(1, input_dim=2))
    m.add(Dense(2))
    m.add(Activation('sigmoid'))
    m.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


m = genModel()
m.fit(posi_ball, y_sim, nb_epoch=200, batch_size=1000, validation_split=0.2)

pass