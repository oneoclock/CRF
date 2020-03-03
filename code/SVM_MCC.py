from liblinear_m.python.liblinearutil import * 
from liblinear_m.python.commonutil import *
from read_data import *
import numpy as np
import matplotlib.pyplot as plt

FILENAME1 = '../data/train.txt'
FILENAME2 = '../data/test.txt'

def svm_mc(FILENAME1,FILENAME2):
    train_data = read_train(FILENAME1)

    test_data = read_train(FILENAME2)
    
    x = []
    y = []
    
    for word in train_data:
        for i in range(word[0].shape[0]):
            x.append(word[0][i])
            y.append(word[1][i])
            
    x_test = []
    y_test = []
    
    for word in test_data:
        for i in range(word[0].shape[0]):
            x_test.append(word[0][i])
            y_test.append(word[1][i])
    
    #training and prediction using liblinear
    prob  = problem(y, x)
    params = []
    params = [parameter('-s 0 -c 100 -B -1'),parameter('-s 0 -c 1000 -B -1'),parameter('-s 0 -c 5000 -B -1'),parameter('-s 0 -c 10000 -B -1')]
    
    p_acc_tests = []
    p_label_tests = []
    for param in params: 
        m = train(prob, param)
        #prediction on train data
        p_label, p_acc, p_val = predict(y, x, m, '-b 1')
        
        #prediction on test data
        p_label_test, p_acc_test, p_val_test = predict(y_test, x_test, m, '-b 1')
        p_label_tests.append(p_label_test)
        p_acc_tests.append(p_acc_test[0])
        
        
    
    #plot
    print(p_acc_tests)
    c=[1,10,100,1000]
    plt.figure()
    plt.xlabel('C')
    plt.ylabel('Accuracy for letter wise predictions on Test Data')
    plt.plot(c,p_acc_tests)
    plt.show()
    
    
    #Calculates word wise accuracy for training data
    start=0
    end=0
    arr=[]
    for word in train_data:
    
        end = end + word[1].shape[0]
        if np.array_equal(word[1],p_label[start:end]):
            arr.append(True)
        else:
            arr.append(False)
        start = end
    
    whole_word_acc=sum(arr)/len(arr)
    print(whole_word_acc)
    
    #Calculates word wise accuracy for test data
    start=0
    end=0
    arr_test=[]
    whole_word_acc_test = []
    for p_label_test_itr in p_label_tests:
        for word in test_data:
        
            end = end + word[1].shape[0]
            if np.array_equal(word[1],p_label_test_itr[start:end]):
                arr_test.append(True)
            else:
                arr_test.append(False)
            start = end
    
        whole_word_acc_test.append(sum(arr_test)/len(arr_test))
    
    print(whole_word_acc_test)
    
    c=[1,10,100,1000]
    plt.figure()
    plt.xlabel('C')
    plt.ylabel('Word wise Accuracy on Test Data')
    plt.plot(c,whole_word_acc_test)
    plt.show()
    
    return

if __name__=='__main__':
    svm_mc(FILENAME1,FILENAME2)
