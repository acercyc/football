import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from football.dataInfo import *
from .dataInfo import *
from numpy import random 





def genUniformDist(nSampleDim, xRange, yRange):
    x = random.uniform(xRange[0], xRange[1], nSampleDim)
    y = random.uniform(yRange[0], yRange[1], nSampleDim)
    return x, y

def genHaifFieldPosition(nSample):
    posi_team1 = genUniformDist( (nSample, 11), (fieldSize_x[0], 0), fieldSize_y )
    posi_team2 = genUniformDist( (nSample, 11), (0, fieldSize_x[1]), fieldSize_y )
    posi_ball = genUniformDist( (nSample, 1), fieldSize_x, fieldSize_y )

    posi_team1 = np.concatenate(posi_team1, 1)
    posi_team2 = np.concatenate(posi_team2, 1)
    posi_ball  = np.concatenate(posi_ball , 1)
    
    return posi_team1, posi_team2, posi_ball