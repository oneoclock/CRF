# -*- coding: utf-8 -*-
#%%
# Optimizing the calculations
import numpy as np
import scipy.optimize as opt
from crfmodel import CRFModel
from read_data import read_train
#%%
def logsumexp_trick(sum_term):
    """
    Perform logsumexp trick to handle numeric overflow/underflow
    """
    max_term = np.max(sum_term)
    return max_term + np.log(np.sum(np.exp(sum_term-max_term)))

#%%
def get_logCRF(train, model):
    """
    CRF for single word and label
    """
    word = train[0]
    Y = train[1]
    char_count, _ = word.shape
    # calculating forward messages
    alpha = np.zeros((char_count, model.dimY))
    first_term = np.dot(word, model.getW(model.labels))
    second_term = model._T
    for i in range(1, char_count):
        sum_term = (first_term[i-1] + alpha[i-1]) + second_term
        alpha[i] = np.apply_along_axis(logsumexp_trick, 1, sum_term) 
    # getting logZ from messages
    logZ = logsumexp_trick(first_term[char_count-1]+alpha[char_count-1])
    w_term = np.sum(model.getW(Y).transpose() * word) # $\sum_{j=1}^m {W_{yj} . x_j}$
    t_term = np.sum(model.getT(Y[:-1], Y[1:])) #$T_{yj, yj+1}
    value = -logZ + w_term + t_term
    return value

def log_crf_wrapper(x, train, dimX, dimY, from_file=False):
    model = CRFModel(dimX, dimY)
    model.load_X(x, from_file=from_file)
    crfs = np.apply_along_axis(get_logCRF, 1, train,*[model])
#    print(x)
#    print(crfs.mean())
    return np.mean(crfs)

#%%
def get_marginals(word, model):
    """
    Calculate the marginals P(y_s|X) and P(y_s, y_s+1|X)
    using forward and backward messages
    returns (mx26, m-1x26x26) marginal distributions for each letter in the word
    """
    # forward and backward message at once
    char_count, _ = word.shape
    alpha = np.zeros((char_count, model.dimY)) # alphas
    lbeta = np.zeros((char_count, model.dimY)) # log version of betas

    first_term = np.dot(word, model.getW(model.labels))
    second_term_a = model._T
    second_term_b = model._T.T
    for i in range(1, char_count):
        sum_term_a = (first_term[i-1] + alpha[i-1]) + second_term_a
        sum_term_b = (first_term[char_count-i] +lbeta[char_count-i]) + second_term_b
        alpha[i] = np.apply_along_axis(logsumexp_trick, 1, sum_term_a) 
        lbeta[char_count-i-1] = np.apply_along_axis(logsumexp_trick, 1, sum_term_b)

    marginal_Y = np.zeros((char_count, model.dimY))
    marginal_Y_Y1 = np.zeros((char_count-1, model.dimY, model.dimY))  
    
    for i in range(char_count):
        sum_term = first_term[i] + alpha[i] + lbeta[i]
        log_marginal_y = sum_term - logsumexp_trick(sum_term)
        marginal_Y[i] = np.exp(log_marginal_y)
        # calculate other marginal dist as well
        if i < char_count-1:
            transition = model._T.transpose() # T_{yi, yi+1}
            outer_sum_w = np.add.outer(first_term[i], first_term[i+1]).reshape(model.dimY,model.dimY)
            outer_sum_m = np.add.outer(alpha[i], lbeta[i+1])
            sum_term_all = outer_sum_w + transition + outer_sum_m
            log_marginal_y_y1 = sum_term_all - logsumexp_trick(sum_term_all)
            marginal_Y_Y1[i] = np.exp(log_marginal_y_y1)
    # Got Denominator same as Zx , which is correct
    return marginal_Y, marginal_Y_Y1

def get_ind(labels, k):
    """
    Indicator func [label == k]
    """
    return (np.array(labels) == k).astype('float64')
    
            
def calculate_gradient_crf(train, model):
    """
    calculate the gradient for given word with m chars and corresponding true labels
    W and T are initial weights
    TODO matrix implementation
    """
    word = train[0]
    label = train[1]
    char_count, _ = word.shape
    # get the marginals
    marY, marY1 = get_marginals(word, model)
    # calculate w_k for all 26 Ws. To do matrix approach
    grad_W = []
    for k in range(1,model.dimY+1):
        ind_k = get_ind(label, k)
        marginal_k = marY[:,k-1]
        grad_k = np.dot((ind_k - marginal_k), word)
        grad_W.append(grad_k)
        
    zero_mat = np.zeros((model.dimY,model.dimY))
    label = np.array(label)
    pairs = list(zip(label[:-1]-1, label[1:]-1))
    grad_T = np.zeros((model.dimY,model.dimY))
    for i,pair in enumerate(pairs):
        ind_ij = zero_mat.copy()
        ind_ij[pair] = 1.0
        grad_T += ind_ij - marY1[i]

    grad_W = np.array(grad_W)
    g = np.concatenate([grad_W.reshape(-1), grad_T.reshape(-1)])
    return g

def grad_crf_wrapper(x, train, dimX, dimY, from_file=False):
#    print(x)
    model = CRFModel(dimX, dimY)
    model.load_X(x, from_file=from_file)
    g = np.apply_along_axis(calculate_gradient_crf, 1, train, *[model])
#    print(g.mean(axis=0))
    return g.mean(axis=0)


#%%
#train_data = read_train("../data/train.txt")
#train = np.array(train_data)
#word_list = train[:,0]
#label_list = train[:,1]
#print("word_list shape :", word_list.shape)
#print("label_list shape :", label_list.shape)
#print("word shape:", word_list[3].shape)
#model = CRFModel(dimX=128, dimY=26)
#x = np.loadtxt("../data/model.txt")
#model.load_X("../data/model.txt", from_file=True)
#word = word_list[0]
#get_logCRF(train[0], model)
#%%
#import time
#start = time.time()
#print(start)
##mean_crf = get_logCRF(train[0], model)
#mean_crf = log_crf_wrapper(x, train, 128, 26, from_file=True)
#g = grad_crf_wrapper(x, train, 128, 26, from_file=True)
#print(time.time() - start)
#print(mean_crf, g)
#%%
def test_check_grad():
    DIMX=5
    DIMY=3
    #def check_grad():
    n_words = 5
    n_chars = 2
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

    
        
    print("CRF = ",log_crf_wrapper(x,train, DIMX, DIMY))
    g = grad_crf_wrapper(x, train, DIMX, DIMY)
    print("g = {}".format(g))
    score = opt.check_grad(log_crf_wrapper, grad_crf_wrapper, x, *[train, DIMX, DIMY])
    print("Score = ",score)
    assert score < 1.0e-4
#test_check_grad()

#%%
#%timeit -n 1 log_crf_wrapper(x, train, 128, 26, from_file=True)

def generate_result():
    train_data = read_train("../data/train.txt")
    train = np.array(train_data)    
    meanLogCRF = log_crf_wrapper("../data/model.txt", train, 128, 26, True)
    print(meanLogCRF) #-29.954718407620692   
    g = grad_crf_wrapper("../data/model.txt", train, 128, 26, from_file=True)
    print(g)
    return g
#generate_result()
#%% 
#train_data = read_train("../data/train.txt")
#train = np.array(train_data)    
#x = np.loadtxt("../data/model.txt")
#t = x[-26*26:].T.reshape(-1)
#x[-26*26:] = t
#score = opt.check_grad(log_crf_wrapper, grad_crf_wrapper, x, *[train, 128, 26])
#print(score)
##%%
#(-29.55662008609797 - (-29.55662008665368 )) / 1.49011612e-08
#%%
#Save in file
def save_grad():
    g = generate_result() 
    gradW = g[:128*26]
    gradT = g[128*26:].reshape(26,26).transpose().reshape(-1) # converting to matlab format
    with open("../result/gradient.txt", 'w') as file:
        length = len(gradT)
        for grad in gradW:
            print(grad, file=file)
        for i,grad in enumerate(gradT):
            if i == length -1:
                print(grad, file=file, end="")
            else:
                print(grad, file=file)
#save_grad()