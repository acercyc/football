import matplotlib.pyplot as plt
import numpy as np
import time
import lib.lib as lib
import pandas as pd
import sys

d = pd.read_csv('valid_data/2016052108_player.csv')
lib.plot_moving(d)

