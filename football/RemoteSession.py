# ---------------------------------------------------------------------------- #
#                                                                              #
#                            Control Remote Session                            #
#                                                                              #
# ---------------------------------------------------------------------------- #
# 1.0 - Acer 2017/01/20 14:27


import os
dispPort = !echo $DISPLAY
os.environ['DISPLAY'] = dispPort[0]