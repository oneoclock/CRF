# -*- coding: utf-8 -*-
#%%
# Calculate objective as well as the gradients
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
def forward_pass(word, model):
    """
    Forward message passing for given word, W and T
    Used to calculate Z_x which is used in log(P(y^t}X*t)) which is used in objective
    Also used to calculate marginal distribution P(ys=y|X^t)
    Z_x is calculated from space of Y^m
    """
    char_count, _ = word.shape
    alpha = np.zeros((char_count, model.dimY))
    first_term = np.dot(word, model.getW(model.labels))
    second_term = model._T
    for i in range(1, char_count):
        sum_term = (first_term[i-1] + alpha[i-1]) + second_term
        alpha[i] = np.apply_along_axis(logsumexp_trick, 1, sum_term) 
    return alpha

def backward_pass(word, model):
    """
    Backward message passing for given word, W and T
    Used to calculate either Z_x or marginal distribution P(y_i|X), P(y_i,y_i+1|X)
    """
    char_count, _ = word.shape
    lbeta = np.zeros((char_count, model.dimY)) # log version of betas
    first_term = np.dot(word, model.getW(model.labels))
    second_term = model._T.T
    for i in reversed(range(0, char_count-1)):
        sum_term = (first_term[i+1] +lbeta[i+1]) + second_term
        lbeta[i] = np.apply_along_axis(logsumexp_trick, 1, sum_term)
    return lbeta

#%%

def get_logZ_back(word, model):
    """
    Get logZ using lbetas
    """
    char_count, _ = word.shape
    labels = tuple(range(1,model.dimY+1))
    lbeta = backward_pass(word, model)
    lbeta_1 = lbeta[0]
    first_term = model.getW(labels).transpose().dot(word[0])
    logZ = logsumexp_trick(first_term + lbeta_1)
    return logZ
    

def get_logZ(word, model):
    """
    Get logZ using alphas
    """
    char_count, _ = word.shape
    labels = tuple(range(1,model.dimY+1))
    alpha = forward_pass(word, model)
    alpha_m = alpha[char_count-1]
    first_term = model.getW(labels).transpose().dot(word[char_count-1])
    logZ = logsumexp_trick(first_term+alpha_m)
    return logZ

def get_logCRF(word, Y, model):
    """
    CRF for single word and label
    """
    logZ = get_logZ(word, model)
    first_term = np.sum(model.getW(Y).transpose() * word) # $\sum_{j=1}^m {W_{yj} . x_j}$
    second_term = np.sum(model.getT(Y[:-1], Y[1:])) #$T_{yj, yj+1}
    value = -logZ + first_term + second_term
    return value

def log_crf_wrapper(x, word_list, label_list, dimX, dimY, from_file=False):
    model = CRFModel(dimX, dimY)
    model.load_X(x, from_file=from_file)
    return get_logCRF_all(model, word_list, label_list)

    

def get_logCRF_all(model, word_list, label_list):
    """
    Takes a list of words with labels and other learnt parameters to return log(P(y^t|X^t))
    averaged for all words
    """
    avg_crf = 0.0
    for i,word in enumerate(word_list):
        Y = label_list[i]
        char_count, _ = word.shape
        labels = tuple(range(1,model.dimY+1))
        alpha = forward_pass(word, model)
        first_term = model.getW(labels).transpose().dot(word[char_count-1])
        logZ = logsumexp_trick(first_term+alpha[char_count-1])       
        sum_wx = np.sum(model.getW(Y).transpose() * word) # $\sum_{j=1}^m {W_{yj} . x_j}$
        sum_t = np.sum(model.getT(Y[:-1], Y[1:])) #$T_{yj, yj+1}
        avg_crf += -logZ + sum_wx + sum_t
    return avg_crf / len(word_list)

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

    marginal_Y = []
    marginal_Y_Y1 = []
    for i in range(char_count):
        inner_i = model.getW(model.labels).transpose().dot(word[i])
        sum_term = inner_i + alpha[i] + lbeta[i]
        log_marginal_y = sum_term - logsumexp_trick(sum_term)
        marginal_Y.append(np.exp(log_marginal_y))
        
        # calculate other marginal dist as well
        if i < char_count-1:
            inner_iplus1 = model.getW(model.labels).transpose().dot(word[i+1]) # Wy_i+1, x_i+1
            alpha_i = alpha[i] # a_i
            lbeta_iplus1 = lbeta[i+1] # b_i+1
            transition = model._T.transpose() # T_{yi, yi+1}
            outer_sum_w = np.add.outer(inner_i, inner_iplus1).reshape(model.dimY,model.dimY)
            outer_sum_m = np.add.outer(alpha_i, lbeta_iplus1)
            sum_term_all = outer_sum_w + transition + outer_sum_m
            log_marginal_y_y1 = sum_term_all - logsumexp_trick(sum_term_all)
#            marginal_y_y1 = np.exp(outer_sum_w + transition + outer_sum_m)
#            marginal_y_y1 = marginal_y_y1 / marginal_y_y1.sum()
            marginal_Y_Y1.append(np.exp(log_marginal_y_y1))
    # Got Denominator same as Zx , which is correct
    return np.array(marginal_Y), np.array(marginal_Y_Y1)

def get_ind(labels, k):
    """
    Indicator func [label == k]
    """
    return (np.array(labels) == k).astype('float64')
    
            
def calculate_gradient_crf(word, label, model):
    """
    calculate the gradient for given word with m chars and corresponding true labels
    W and T are initial weights
    TODO matrix implementation
    """
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


def grad_crf_wrapper(x, word_list, label_list, dimX, dimY, from_file=False):
#    print(x)
    model = CRFModel(dimX, dimY)
    model.load_X(x, from_file=from_file)
#    T[:,:] = 0.0
    avg = np.zeros(x.shape)
    for i,word in enumerate(word_list):
        label = label_list[i]
        avg += calculate_gradient_crf(word, label, model)
    g = avg/len(word_list)
#    g[-DIMY*DIMY:] = 0
    return g


#%%
# TODO Generate the score and result files
def generate_result():
    train_data = read_train("../data/train.txt")
    train = np.array(train_data)
    word_list = train[:,0]
    label_list = train[:,1]
    print("word_list shape :", word_list.shape)
    print("label_list shape :", label_list.shape)
    print("word shape:", word_list[3].shape)
    
    model = CRFModel(dimX=128, dimY=26)
    model.load_X("../data/model.txt", from_file=True)
    print(model._T.shape)
    print(model._W.shape)
    meanLogCRF = get_logCRF_all(model, word_list, label_list)
    print(meanLogCRF) #-29.954718407620692   
    
    x = np.loadtxt("../data/model.txt")
    g = grad_crf_wrapper(x, word_list, label_list, 128, 26, from_file=True)
    return g
#generate_result()
#%%    
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
#%%
#x = np.loadtxt("../data/model.txt")
#x.shape  
#model = CRFModel(dimX=128, dimY=26)
#model.load_X("../data/model.txt", from_file=True)
##print(x[-26*26:][0:26])
##print(model._T[0])
#
#W1 = model._W
#W1.shape
#norm_sq = np.dot(W1.transpose(), W1)
#norm_sq.sum()
#%%
#print(np.sum(W1*W1))

#%%
# Test Case
def test_check_grad():
    DIMX=5
    DIMY=3
    #def check_grad():
    n_words = 5
    n_chars = 2
    np.random.seed(3)
    word_list = np.random.randint(10,size=DIMX*n_chars*n_words).reshape(n_words,n_chars,DIMX)
    label_list = np.random.choice(range(1,DIMY+1),size=n_chars*n_words).reshape(n_words,n_chars)
#    x = np.zeros((DIMX*DIMY)+DIMY*DIMY)
    x= np.random.uniform(size=(DIMX*DIMY)+(DIMY*DIMY))
    model = CRFModel(DIMX, DIMY)
    model.load_X(x)
    W1, T1 = model._W, model._T
    print("word = ",word_list.shape)
    print("W = ",W1.shape)
    print("T = ",T1.shape)
    print("label = ",label_list.shape)
    print("CRF = ",log_crf_wrapper(x, word_list, label_list, DIMX, DIMY))
    g = grad_crf_wrapper(x, word_list, label_list, DIMX, DIMY)
    print("g = {}".format(g))
    score = opt.check_grad(log_crf_wrapper, grad_crf_wrapper, x, *[word_list, label_list, DIMX, DIMY])
    print("Score = ",score)
    assert score < 1.0e-4
#TEMPtest_check_grad()

#%%
#(-1.3862943611198906 - (-1.3862943648451809)) / 1.49011612e-08
## Mean Log CRF = -29.954718407620692
#train_data = read_train("../data/train.txt")
#train = np.array(train_data)
#word_list = train[:,0]
#label_list = train[:,1]
#print("word_list shape :", word_list.shape)
#print("label_list shape :", label_list.shape)
#print("word shape:", word_list[3].shape)
#model = CRFModel(dimX=128, dimY=26)
#model.load_X("../data/model.txt", from_file=True)
#print(model._T.shape)
#print(model._W.shape)
#word = word_list[0]
#label = label_list[0]
##get_logCRF(word, label, model)
#alpha1 = forward_pass(word, model)
#beta1 = backward_pass(word, model)
#alpha2, beta2 = get_marginals(word, model)
#np.allclose(alpha1, alpha2)
#np.allclose(beta1, beta2)
#%%

#%timeit -n 10 get_logCRF(word, label, model)
#logsumexp_trick(first_term[i-1] + second_term[:,1] + alpha[i-1])
    
#    for y1 in model.labels: # for each next_label
#        second_term = model.getT(model.labels, [y1]*model.dimY) # $T_{yi-1, yi}$
#        sum_term = first_term + second_term + alpha[i-1]
#        alpha[i,y1-1] = logsumexp_trick(sum_term)
#alpha
#model._T
#model.getT(model.labels,1)
