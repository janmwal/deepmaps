# source : https://stackoverflow.com/questions/20818121/save-csv-to-mat-or-binary

import pandas as pd 
import sys
import scipy.io

if __name__ == "__main__":
    
    print("Script doesn't work properly")
    sys.exit(-1)

    if len(sys.argv) != 3:
        print ("Usage: %s input.mat output.csv" % sys.argv[0])
        sys.exit(-1)

    with open(sys.argv[1]) as f:
        matfile = scipy.io.loadmat(f)
    
    df = pd.DataFram(matfile)
    df.to_csv(sys.argv[2])