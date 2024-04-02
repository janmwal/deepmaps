# source : https://stackoverflow.com/questions/20818121/save-csv-to-mat-or-binary

import csv  
import sys
import numpy
import scipy.io

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage: %s input.csv output.mat" % sys.argv[0])
        sys.exit(-1)

    data = [ ]
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)
        for row in reader:
            rowData = [ float(elem) for elem in row ]
            data.append(rowData)

    matrix = numpy.array(data)
    scipy.io.savemat(sys.argv[2], {'csvmatrix':matrix})