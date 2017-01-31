# ============================================================================ #
#                                                                              #
#                         For football data processing                         #
#                                                                              #
# ============================================================================ #
# 1.0 - Acer 2017/01/17 16:54


import tarfile
import pandas as pd
import numpy as np
import sys
from . import dataInfo
import matplotlib.pyplot as plt
import os
import re


# ---------------------------------------------------------------------------- #
#                                                                              #
#                                   Read Data                                  #
#                                                                              #
# ---------------------------------------------------------------------------- #

def read_csv_tar(iFile):
    tar = tarfile.open(iFile)
    files = tar.getmembers()
    f = tar.extractfile(files[3])
    df = pd.read_csv(f)

    return df


def read_csv_by_index(iFile):
    fNames = os.listdir('valid_data')
    iFName = [True if re.search('\d+.csv', x) else False for x in fNames]
    fNames = np.array(fNames)[np.array(iFName)]
    fNames = fNames[np.array(iFile)]

    d = pd.DataFrame()
    for fName in fNames:
        print(fName)
        d = d.append(pd.read_csv('valid_data/' + fName))

    return d


# ---------------------------------------------------------------------------- #
#                                                                              #
#                                Data arrangement                              #
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# parse the player column into 22 columns for 22 players 
# ---------------------------------------------------------------------------- #
def parsePlayersCol(df):
    d_player = df['Players']
    d_player_df = d_player.str.split(':').apply(pd.Series)
    d_player_df.columns = ['pCol_' + str(col) for col in d_player_df.columns]
    df = pd.concat([df, d_player_df], axis=1)
    df = df.drop(['Players'], 1)

    return df


# ---------------------------------------------------------------------------- #
# add player based data columns to the df
# ---------------------------------------------------------------------------- #
# 1.0 - Acer 2017/01/18 11:50
def addPlayerBasedColumn(df):
    # Parse player column
    df = parsePlayersCol(df)

    # find unique player number
    playerSet = set(df.loc[:, 'pCol_0':'pCol_21'].unstack())
    playerSet = sorted(playerSet)

    # create empty df for re-mapping players and data 
    playerX = pd.DataFrame(index=range(df.shape[0]), columns=playerSet)
    playerY = playerX.copy()

    # sort columns by name
    playerX = playerX.reindex_axis(sorted(playerX.columns), axis=1)
    playerY = playerY.reindex_axis(sorted(playerY.columns), axis=1)

    # loop to visit all time row and players 
    for pName in playerSet:
        sys.stdout.write("%s\n" % pName)
        for i in range(df.shape[0]):

            # extract player column from the current row
            l = df.loc[i, 'pCol_0':'pCol_21']

            # find target player column as target column 
            iPName = np.where(l == pName)[0]

            # extract xy data based on target column and save it in new dataframe
            if iPName.size == 0:
                pass
            else:
                x = df.loc[i, 'H1X':'A11X']
                y = df.loc[i, 'H1Y':'A11Y']
                playerX.loc[[i], pName] = float(x[iPName])
                playerY.loc[[i], pName] = float(y[iPName])

    # change column name
    ## shorten names
    playerX = playerX.rename(columns=lambda x: x.replace('away', 'A'))
    playerX = playerX.rename(columns=lambda x: x.replace('home', 'H'))

    playerY = playerY.rename(columns=lambda x: x.replace('away', 'A'))
    playerY = playerY.rename(columns=lambda x: x.replace('home', 'H'))

    ## add player_x_
    playerX.columns = ['player_X_' + playerX.columns]
    playerY.columns = ['player_Y_' + playerY.columns]

    # combine with the original df
    df = df.join(playerX).join(playerY)

    return df


def addIsNewSession(d):
    """ 
    detect large moving distance and indicate it as the new session start 
    1.0 - Acer 2017/01/20 17:38
    """
    
    d = d.copy()
    d_posi = d.ix[:, 'H1X':'A11Y']
    
    posi_diff = np.sum( abs(np.diff(d_posi, axis = 0)), axis=1 )
    iNewSession = np.array((posi_diff > 1000).nonzero())[0]+1
    iNewSession = np.insert(iNewSession, 0, 0)

    d['isNewSession'] = 0
    d.ix[iNewSession, 'isNewSession'] = 1
    
    return d


def addSideChangeDist(d):
    """
    create a side change column
    a column indicate how many frame will be a possession side change 
    1.0 - Acer 2017/01/23 14:59
    """
    
    # reverse possession order
    poss_rev = d.ix[::-1,'Possession_A'].values

    # change 0 to -1
    poss_rev[poss_rev==0] = -1

    # add temporal distance value
    iFlip = np.diff(poss_rev).nonzero()[0] + 1 
    printer = 0;
    sideChangeDist = np.empty_like(poss_rev)
    for i in range(len(poss_rev)):
        if i in iFlip:
            printer = 0
        printer = printer + poss_rev[i]
        sideChangeDist[i] = printer
    
    # reverse back
    sideChangeDist = sideChangeDist[::-1] 

    d['sideChangeDist_by_A'] = sideChangeDist
    
    return d


# ---------------------------------------------------------------------------- #
#                                                                              #
#                                    Plotting                                  #
#                                                                              #
# ---------------------------------------------------------------------------- #

# 1.0 - Acer 2017/01/19 12:20
def plot_moving(d, timeStep=5):
    X = d.loc[:, 'H1X':'A11X']
    Y = d.loc[:, 'H1Y':'A11Y']

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(0, d.shape[0], timeStep):
        # team H
        x = X.ix[i, 0:11]
        y = Y.ix[i, 0:11]
        ax.plot(x, y, 'ro')

        # team A
        x = X.ix[i, 11:22]
        y = Y.ix[i, 11:22]
        ax.plot(x, y, 'bo')

        # ball
        ax.plot(d['BX'][i], d['BY'][i], 'ks', markersize=(d['BZ'][i]) / 5 + 10, fillstyle='none')
        ax.plot(d['BX'][i], d['BY'][i], 'k^', markersize=5, fillstyle='none')

        ax.set_xlim(dataInfo.fieldSize_x)
        ax.set_ylim(dataInfo.fieldSize_y)

        fig.canvas.draw()
        ax.cla()


def plot_moving_playerBased(d, timeStep=5, isPlotSessionBreak = False):
    """ Player based moving presentation """
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if isPlotSessionBreak:
        ns = d.isNewSession
        ns_t = ns
        for i in range(100):
            ns_t = np.roll(ns_t, -1)
            ns = ns + ns_t * (i+1)

    for i in range(0, d.shape[0], timeStep):
        # team H
        x = d.loc[i, d.columns.str.contains('player_X_H')]
        y = d.loc[i, d.columns.str.contains('player_Y_H')]
        ax.plot(x, y, 'ro')

        # team A
        x = d.loc[i, d.columns.str.contains('player_X_A')]
        y = d.loc[i, d.columns.str.contains('player_Y_A')]
        ax.plot(x, y, 'bo')

        # ball
        ax.plot(d['BX'][i], d['BY'][i], 'ks', markersize=(d['BZ'][i]) / 5 + 10, fillstyle='none')
        ax.plot(d['BX'][i], d['BY'][i], 'k^', markersize=5, fillstyle='none')
        
        #raise Exception('sss')
        
        if isPlotSessionBreak:
            if ns[i] > 0:
                ax.plot(np.array([dataInfo.fieldSize_x[0], dataInfo.fieldSize_x[0] + ns[i]*30]) + 10,
                        np.array([dataInfo.fieldSize_y[0], dataInfo.fieldSize_y[0]]) + 10,
                        'g', linewidth = 10)

        ax.set_xlim(dataInfo.fieldSize_x)
        ax.set_ylim(dataInfo.fieldSize_y)

        fig.canvas.draw()
        ax.cla()
        
        
        