import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import football.lib as fb
import football.dataGenerator as gen

import football.shelve_ext as she

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.utils.np_utils import to_categorical


# --------------------------------- Read data --------------------------------
d = she.load('temp_20170130_193524', 'd')


# d = pd.read_csv('valid_data/2016022707_player.csv')
# d = fb.read_csv_by_index(range(10))


def extractData(d):
    X_posi = d.ix[:, 'H1X':'A11Y']
    X_ball = d.ix[:, 'BX':'BZ']
    X = pd.concat([X_posi, X_ball], axis=1)

    y = d.ix[:, 'Possessvpion_A']

    X = X / fb.dataInfo.fieldSize_x[1]
    y = to_categorical(y)

    return X, y


def extractDataAndGroup(d):
    X_posi = d.ix[:, 'H1X':'A11Y']
    X_ball = d.ix[:, 'BX':'BZ']
    X = pd.concat([X_posi, X_ball], axis=1)

    y = d.ix[:, 'Possession_A']

    X = X / fb.dataInfo.fieldSize_x[1]
    y = to_categorical(y)

    X = np.array(X)

    X_team1 = X[:, np.hstack((range(0, 11), range(22, 33)))]
    X_team2 = X[:, np.hstack((range(11, 22), range(33, 44)))]
    X_ball = X[:, -3:]

    return ([X_team1, X_team2, X_ball], y)


X, y = extractDataAndGroup(d)


# -------------------------------- Build model -------------------------------
def genModel():
    m_team1 = Sequential()
    m_team1.add(Dense(11, input_dim=22))
    m_team1.add(BatchNormalization())
    m_team1.add(Activation('sigmoid'))
    m_team1.add(Dropout(0.5))

    m_team2 = Sequential()
    m_team2.add(Dense(11, input_dim=22))
    m_team2.add(BatchNormalization())
    m_team2.add(Activation('sigmoid'))
    m_team2.add(Dropout(0.5))

    m_ball = Sequential()
    m_ball.add(Dense(2, input_dim=3))
    m_ball.add(BatchNormalization())
    m_ball.add(Activation('sigmoid'))
    m_ball.add(Dropout(0.5))

    m_inputMerge = Merge([m_team1, m_team2, m_ball], mode='concat')

    m = Sequential()
    m.add(m_inputMerge)
    m.add(Dense(64))
    m.add(BatchNormalization())
    m.add(Activation('sigmoid'))
    m.add(Dropout(0.5))

    m.add(Dense(64))
    m.add(BatchNormalization())
    m.add(Activation('sigmoid'))
    m.add(Dropout(0.5))

    m.add(Dense(64))
    m.add(BatchNormalization())
    m.add(Activation('sigmoid'))
    m.add(Dropout(0.5))

    m.add(Dense(2))
    m.add(BatchNormalization())
    m.add(Activation('sigmoid'))

    m.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


m = genModel()
history = m.fit(X, y, nb_epoch=40, batch_size=1000, validation_split=0.2)

# ----------------------------- Load testing data ----------------------------

d_test = fb.read_csv_by_index(range(10, 13))
X_test, y_test = extractDataAndGroup(d_test)
result = m.evaluate(X_test, y_test)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()