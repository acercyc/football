# 1.0 - Acer 2017/01/19 17:25

import football.lib as fb
import pandas as pd

d = pd.read_csv('valid_data/2016022707_player.csv')
fb.plot_moving_playerBased(d)
