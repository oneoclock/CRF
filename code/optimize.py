# -*- coding: utf-8 -*-
#%%
import numpy as np
import scipy.optimize as opt
from crfmodel import CRFModel
from gradient import grad_crf_wrapper, log_crf_wrapper 
from decoder import crf_decode
from read_data import read_train

#%%

def compare(prediction, true_label):
    """
    """
    letterCount = 0.0
    wordCount = len(prediction)
    print(wordCount)
    letterMatch = 0.0
    wordMatch = 0.0
    wordArrMatch = 0.0
    for i in range(wordCount):
        wordPred = prediction[i]
        wordTrue = true_label[i]
        if np.array_equal(np.array(wordPred), np.array(wordTrue)):
            wordArrMatch+=1
        matchCount = 0.0
        for j,pred in enumerate(wordPred):
            letterCount += 1
            if pred == wordTrue[j]:
                letterMatch += 1
                matchCount += 1
        if matchCount == len(wordPred):
            wordMatch +=1
    return letterMatch/letterCount, wordArrMatch/wordCount
            

#%%
def test_log_crf(x, train, c, dimX, dimY, from_file=True):
    logCrf = log_crf_wrapper(x, train, dimX, dimY, from_file=from_file)
    model = CRFModel(dimX, dimY)
#    print(x)
    model.load_X(x,from_file=from_file)
    W = model._W # column format
    T = model._T # column format
    # Compute the objective value of CRF
    f = (-c *logCrf)  + 0.5 * np.sum(W*W) + 0.5 * np.sum(T*T) # objective log-likelihood + regularizer
#    print(f)
    return f

def test_grad_crf(x, train, c, dimX, dimY, from_file=True):
    model = CRFModel(dimX, dimY)
    model.load_X(x,from_file=from_file)
    W = model._W # column format
    T = model._T # column format
    reg = np.concatenate([W.T.reshape(-1), T.T.reshape(-1)])
    g = grad_crf_wrapper(x, train, dimX, dimY, from_file=from_file)
    g = -c * g + reg
    return g
#%%
def test_check_grad():
    DIMX=12
    DIMY=5
    #def check_grad():
    n_words = 10
    n_chars = 3
    np.random.seed(3)
    word_list = np.random.randint(10,size=DIMX*n_chars*n_words).reshape(n_words,n_chars,DIMX).tolist()
    label_list = np.random.choice(range(1,DIMY+1),size=n_chars*n_words).reshape(n_words,n_chars).tolist()
    x = np.zeros((DIMX*DIMY)+DIMY*DIMY)
#    x= np.random.uniform(size=(DIMX*DIMY)+(DIMY*DIMY))
    model = CRFModel(DIMX, DIMY)
    model.load_X(x)
    W1, T1 = model._W, model._T
    print("W = ",W1.shape)
    print("T = ",T1.shape)
    train = np.zeros((n_words, 2), dtype='object')
    for i in range(n_words):
        tempX = []
        for word in word_list[i]:
            tempX.append(np.array(word, dtype=float))
        train[i][0] = np.array(tempX)
        train[i][1] = np.array(label_list[i], dtype=int)
        
    score = opt.check_grad(log_crf_wrapper, grad_crf_wrapper, x, *[train, DIMX, DIMY, False])
    print("Score = ",score)
    assert score < 1.0e-3
    score = opt.check_grad(test_log_crf, test_grad_crf, x, *[train, 1000, DIMX, DIMY, False])
    print("Score = ",score)
    assert score < 1.0e-3
test_check_grad()
#%%   
(2079.4415416798356   - 8.855710458252254)/1.49011612e-08
#%%
def crf_obj(x, train_data, c):
    """Compute the CRF objective and gradient on the list of words (word_list)
    evaluated at the current model x (w_y and T, stored as a vector)
    """
    print("Evaluating grad")
    global iteration
    iteration +=1
    print(iteration)
    # x is a vector as required by the solver.
    logCrf = log_crf_wrapper(x,train_data, 128, 26, from_file=False)
    model = CRFModel(128, 26)
    model.load_X(x,from_file=False)
    W = model._W # column format
    T = model._T # column format
    # Compute the objective value of CRF
    f = (-c *logCrf)  + (0.5 * np.sum(W*W)) + (0.5 * np.sum(T*T)) # objective log-likelihood + regularizer
    reg = np.concatenate([W.T.reshape(-1), T.T.reshape(-1)])
    g = grad_crf_wrapper(x, train_data, 128, 26, from_file=False)
    g = -c * g + reg
    return [f, g]
#%%
def crf_test(x, test_data):
    """
    Compute the test accuracy on the list of words (word_list); x is the
    current model (w_y and T, stored as a vector)
    """
    word_list = test_data[:,0]
    true_label = test_data[:,1]
    # x is a vector. so reshape it into w_y and T
    model = CRFModel(128, 26)
    model.load_X(x, from_file=False) # Assume x in the format received from file
    # Compute the CRF prediction of test data using W and T
    y_predict = crf_decode(model, word_list)
#    print(y_predict)

    # Compute the test accuracy by comparing the prediction with the ground truth
    letterAcc, wordAcc = compare(y_predict, true_label)
    print('Letter Accuracy = {}\n'.format(letterAcc))
    print('Word Accuracy = {} \n'.format(wordAcc))
    return y_predict

#%%
def crf_optimize(train_data, test_data, c):
    print('Training CRF ... c = {} \n'.format(c))

    # Initial value of the parameters W and T, stored in a vector
    x0 = np.zeros(128*26+26**2)
#    x0 = np.loadtxt('wtbkc1000.txt')

    result = opt.fmin_l_bfgs_b(crf_obj, x0, args = [train_data, c], maxfun=100,
                           iprint=5)
#    result = opt.fmin_bfgs(test_log_crf, x0, fprime=test_grad_crf,
#        args=(train_data, c, 128, 26, False),
#        maxiter=20, full_output=True,disp=True)
#    result = opt.fmin_tnc(crf_obj, x0, args = [train_data, c], maxfun=100,
#                          ftol=1e-3, disp=5)
    
    model  = result[0]          # model is the solution returned by the opitimizer
    print(result)
    return model, result

#%%
global iteration
iteration = 0
train_data =read_train("../data/train.txt")
train_data = np.array(train_data)
train_data.shape

test_data = read_train("../data/test.txt")
test_data = np.array(test_data)
test_data.shape
#wt, result = crf_optimize(train_data, test_data, 1000)
#accuracy = crf_test(model, test_data)
#print('CRF test accuracy for c = {}: {}'.format(c, accuracy))
#wt
#%%

wt = np.loadtxt('../result/solution.txt')
t = wt[-26*26:].T.reshape(-1)
wt[-26*26:] = t
#crf_obj(wt, train_data, 1000)
preds = crf_test(wt, test_data)

#%%
def generate_prediction(preds):
    with open("../result/prediction.txt", 'w') as file:
        for pred in preds:
            for label in pred:
                print(label, file=file)
generate_prediction(preds)
        
            
#%%
def convert_column_major():
    wt = np.loadtxt('../backup/wtc1000v2.txt')
    t = wt[-26*26:].T.reshape(-1)
    wt[-26*26:] = t
    np.savetxt("../result/solution.txt", wt)
convert_column_major()

#%%