# -*- coding: utf-8 -*-
from liblinear_m.python.liblinearutil import * 
from liblinear_m.python.commonutil import *
from read_data import read_train
import cv2
import numpy as np
import matplotlib.pyplot as plt
from optimize import *
FILENAME1 = '../data/train.txt'
FILENAME2 = '../data/test.txt'

file = open('../data/transform.txt', "r")
file_data= file.read()
file_data = file_data.split("\n")
file.close()

#----------SVM-MC---------------

def svm_mc(dist):
    train_data = read_train(FILENAME1)

    test_data = read_train(FILENAME2)
    
    x = []
    y = []
    
    #for word in train_data:
    #    for i in range(word[0].shape[0]):
    #       x.append(word[0][i])
    #       y.append(word[1][i])
    train1=np.array(read_train(FILENAME1))
    X=train1[:,0]
    Y=train1[:,1]        
    for i in range(X.shape[0]):
        X[i]=X[i].reshape((X[i].shape[0],16,8))    
    #Applying distortion on train data
    #plt.imshow(X[2551][3])
    for trans in file_data[:dist]:
        trans=trans.split(' ')
        for i in range(len(X[int(trans[1])-1])):
            if(trans[0]=='r'):
                t=cv2.getRotationMatrix2D((4,8),-1*int(trans[2]),1)
                X[int(trans[1])-1][i]=cv2.warpAffine(X[int(trans[1])-1][i],t,(8,16))
            elif(trans[0]=='t'):
                t=np.float32([ [1,0,trans[2]], [0,1,trans[3]] ])
                X[int(trans[1])-1][i]=cv2.warpAffine(X[int(trans[1])-1][i],t,(8,16))
    for i in range(X.shape[0]):
        for j in range(X[i].shape[0]):
            x.append(X[i][j].flatten())
            y.append(Y[i][j])
    #print(x[1][1:10])
         
    x_test = []
    y_test = []
    
    for word in test_data:
        for i in range(word[0].shape[0]):
            x_test.append(word[0][i])
            y_test.append(word[1][i])
    
    #training and prediction using liblinear
    prob  = problem(y, x)
    params = []
    params = [parameter('-s 0 -c 100 -B -1')]#,parameter('-s 0 -c 10 -B -1'),parameter('-s 0 -c 100 -B -1'),parameter('-s 0 -c 1000 -B -1')]
    
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
    '''
    c=[1,10,100,1000]
    plt.figure()
    plt.xlabel('C')
    plt.ylabel('Accuracy for letter wise predictions on Test Data')
    plt.plot(c,p_acc_tests)
    plt.show()
    '''
    
    
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
    
    return [p_acc_tests,whole_word_acc_test]

def eval_svm():
    
    tr=[0,500,1000,1500,2000]    
    
###------------SVM evaluate--------------####
    svm_let=np.zeros(5)
    svm_word=np.zeros(5)
    
    
    for i in range(len(tr)):    
        res=svm_mc(tr[i])
        svm_let[i]=res[0][0]
        svm_word[i]=res[1][0]
        print("SVM Accuracy per letter:")
        print(svm_let)
        
        print('SVM Accuracy per word')
        print(svm_word)
    #PLot for SVM
    '''
    plt.figure(1)
    plt.plot(tr,svm_let)
    plt.xlabel('x')
    plt.ylabel('SVM Accuracy per letter')

    plt.figure(2)
    plt.plot(tr,svm_word)
    plt.xlabel('x')
    plt.ylabel('SVM Accuracy per word')
    '''
    
    return svm_let, svm_word


###----------CRF Eval-----------___###
def eval_crf():
    tr=[0,500,1000,1500,2000] 
    crf_let=np.zeros(5)
    
    c=10
    
    for j in range(len(tr)):    
        #LOad data
        train1=np.array(read_train(FILENAME1))
        
        test_data = np.array(read_train(FILENAME2))
        
        X=train1[:,0]
        Y=train1[:,1] 
        
        train2=np.array(train1)
        #applying transformations
        #Reshape to image
        for i in range(X.shape[0]):
            X[i]=X[i].reshape((X[i].shape[0],16,8))
              
        for trans in file_data[:tr[j]]:
            trans=trans.split(' ')
            #Apply the transformations
            for i in range(len(X[int(trans[1])-1])):
                if(trans[0]=='r'):
                    t=cv2.getRotationMatrix2D((4,8),-1*int(trans[2]),1)
                    X[int(trans[1])-1][i]=cv2.warpAffine(X[int(trans[1])-1][i],t,(8,16))
                elif(trans[0]=='t'):
                    t=np.float32([ [1,0,trans[2]], [0,1,trans[3]] ])
                    X[int(trans[1])-1][i]=cv2.warpAffine(X[int(trans[1])-1][i],t,(8,16))
        
        #Flatten to vector
        for i in range(X.shape[0]):
            X[i]=X[i].reshape((X[i].shape[0],128))
            
        train2[:,0]=X
        #train3=train2[:20]
        #test3=test_data[:20]
        w=crf_optimize(train2,test_data,1000)
            
        crf_let[j]=crf_test(w,test_data)
        print(crf_let[j])    
    return crf_let
    '''
    plt.figure(3)
    plt.plot(tr,crf_let)
    plt.xlabel('x')
    plt.ylabel('CRF Accuracy per letter')
    '''
def res():
    tr=[0,500,1000,1500,2000]
    svm_let,svm_word=eval_svm()
    crf_let=eval_crf()
    #crf_let=np.load('results.npy')[2]
    plt.figure(1)
    plt.plot(tr,svm_let)
    plt.plot(tr,crf_let)
    plt.legend(['SVM-MC','CRF'])
    plt.xlabel('x')
    plt.ylabel('Accuracy per letter')

    plt.figure(2)
    plt.plot(tr,svm_word)
    plt.xlabel('x')
    plt.ylabel('SVM Accuracy per word')
    
    #np.save('results.npy',(svm_let,svm_word,crf_let))
    
res()
#-------------------------------
'''class Distort():
    def apply(self,x):
        train=np.array(read_train(FILENAME1))
        X=train[:,0]
        Y=train[:,1]        

        for i in range(X.shape[0]):
            X[i]=X[i].reshape((X[i].shape[0],16,8))
        #plt.imshow(X[2551][3])
        for trans in file_data[:x]:
            trans=trans.split(' ')
            for i in range(len(X[int(trans[1])])):
                if(trans[0]=='r'):
                    t=cv2.getRotationMatrix2D((4,8),-1*int(trans[2]),1)
                    X[int(trans[1])][i]=cv2.warpAffine(X[int(trans[1])][i],t,(8,16))
        plt.imshow(X[2551][3])       
        
'''   

                
#e=Distort()
#e.apply(5)                
