# ---------------------------------------------------------------------------- #
#                                                                              #
#    For examining possession just based on distance between player and ball   #
#                  using nearest neighborhood classification                   #
#                                                                              #
# ---------------------------------------------------------------------------- #
# 1.0 - Acer 2017/01/31 17:11

import numpy as np
import pandas as pd
import football.lib as fb


# read data
d_test = fb.read_csv_by_index(range(10, 13))
X = d_test.ix[:, 'H1X':'A11X']
X = d_test.ix[:, 'H1X':'A11X']
Y = d_test.ix[:, 'H1Y':'A11Y']
X_b = d_test.ix[:, 'BX']
Y_b = d_test.ix[:, 'BY']

# compute square distance
X_diff = np.array(np.expand_dims(X_b, 1)) - np.array(X)
Y_diff = np.array(np.expand_dims(Y_b, 1)) - np.array(Y)
dist = X_diff ** 2 + Y_diff ** 2

# identify nearest player
iMin = np.argmin(dist, 1)


# predict
prediction = iMin <= 10
ans = np.array(d_test.ix[:, 'Possession_H']).astype(bool)

# evaluation
result = np.mean(ans == prediction)
print(result)

# about 75%