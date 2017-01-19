import pandas as pd
import lib.lib as lib
import glob
files = glob.glob('valid_data/*[0-9].csv')

for fName in files:
    print(fName)
    df = pd.read_csv(fName)
    df = lib.addPlayerBasedColumn(df)
    df.to_csv( fName.replace('.csv', '_player.csv') )