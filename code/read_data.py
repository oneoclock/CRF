

import string 
import numpy as np

def read_train(filename):
    mapping = list(enumerate(string.ascii_lowercase))
    mapping = {i[1]: i[0]+1 for i in mapping}

    file = open(filename, "r")
    file_data= file.read()
    file_data = file_data.split("\n")
    file.close()
    
    X, Y, tempX, tempY = [], [], [], []
    for col in file_data[:-1]:
        col = col.split(" ")
        tempY.append(mapping[col[1]])
        tempX.append(np.array(col[5:], dtype=float))
        if int(col[2]) == -1:
            X.append(np.array(tempX))
            Y.append(np.array(tempY, dtype=int))
            tempX.clear()
            tempY.clear()
        else:
            pass

    XY_zip = zip(X,Y)
    return list(XY_zip)

#train_data = read_train('train.txt')
#
#print(train_data[-1])
