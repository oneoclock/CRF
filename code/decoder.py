#%%
import numpy as np
import itertools
from crfmodel import CRFModel

DIMX = 128
DIMY = 26

def parse_decode(input_file, char_count=100, DIMX=128, DIMY=26):
    """
    Takes input path to construct X,W,T matrices
    """
    content = np.loadtxt(input_file)
    x_len = DIMX * char_count # X_1' to X_m'
    X = content[:x_len].reshape(char_count, DIMX) # For python load row-major
    w_len = x_len + (DIMY * DIMX)
    W = content[x_len:w_len].reshape(DIMY, DIMX) # Again row-major order
    T = content[w_len:].reshape(DIMY,DIMY)
    return X, W.transpose(), T
    

#%%
# $argmax_{y \in Y^m}\{\sum_{j=1}^m {W_{yj} . x_j} + \sum_{j=1}^{m-1} T_{yj, yj+1}\}$

#Brute Force Implementation


def crf_decode_bf(model, word_list):
    """
    Brute Force Implementation of CRF Decoding.
    W and T are learnt node and edge potential weights
    word_list Nxmx128 dimensional word vector
    """
    #Looping for now, think about doing as matrix manipulation or batch_size
    predictions = []
    for word in word_list:
        char_count,_ = word.shape
        if char_count > 3:
            raise Exception("Brute Force Algorithm Implementation only limited to 3 characters word")
        # generating all possibilities
        labels = tuple(range(1,model.dimY+1))
        possible = [labels]*char_count
        Y = list(itertools.product(*possible))
        max_term = None
        max_ind = None
        for i,y in enumerate(Y):
            first_term = np.sum(model.getW(y).transpose() * word) # $\sum_{j=1}^m {W_{yj} . x_j}$
            second_term = np.sum(model.getT(y[:-1], y[1:])) #$T_{yj, yj+1}
            sum_term = first_term + second_term
            if max_term is None or max_term <= sum_term:
                max_term = sum_term
                max_ind = i
        predictions.append(Y[max_ind])
    return predictions

#%%

# Dynamic Programming implementation
def crf_decode(model, word_list):
    """
    Dynamic Programming implementation of CRF Decoding.
    """
    predictions = []
    # looping for now. Look for better alternatives
    for word in word_list:
        char_count,_ = word.shape
        labels = tuple(range(1,model.dimY+1))
        # Construct lookup
        lookup = np.zeros((char_count, model.dimY))
        
        for i in range(1,char_count): # dot product only upto second last element
            first_term = model.getW(labels).transpose().dot(word[i-1]) # $W_{yi-1} . X_{i-1}$
            for y1 in labels: # for each next_label
                second_term = model.getT(labels, [y1]*model.dimY) # $T_{yi-1, yi}$
                sum_term = first_term + second_term + lookup[i-1]
                lookup[i,y1-1] = np.max(sum_term) # get best score by all possible current_label
        
        # BackTrack to get the solution
        previousAns = [None]
        score = 0
        for i in reversed(range(0,char_count)): # go from last to first
            first_term = model.getW(labels).transpose().dot(word[i])
            if previousAns[-1] is None: #last label does not have next_label
                score = np.max(first_term + lookup[i]) # lookup contains best score 
#                print(score) # Score to be reported
                ans = np.argmax(first_term + lookup[i])
                previousAns[0] = ans+1 # index to label
            else:
#                second_term = model.getT(labels, [previousAns[-1]]*26)
                second_term = model._T[previousAns[-1]-1]
                ans = np.argmax(first_term+second_term+lookup[i])
                previousAns.append(ans+1)
                
        previousAns.reverse()
        predictions.append(tuple(previousAns))
    return predictions   

#%%
def test():
    X, W, T = parse_decode('../data/decode_input.txt')
    print("X: {} \nW: {}\nT: {}".format(X.shape, W.shape, T.shape))
    ## Test with sample
    model = CRFModel(128, 26)
    model.load_WT(W, T)
    word_list = X[30:42].reshape(-1,3,128)
    pred_bf = crf_decode_bf(model, word_list)
    pred_dp = crf_decode(model, word_list)
    print(pred_bf)
    print(pred_dp)
    assert pred_bf == pred_dp
    generate_result(X, model)
#%%
    
def generate_result(X,model):
    preds = crf_decode(model, X.reshape(-1,100,128))
    with open("../result/decode_output.txt", 'w') as file:
        length = len(preds[0])
        for i,char in enumerate(preds[0]):
            if i == length-1:
                print(char, file=file, end="") #Making sure 100 lines only
            else:
                print(char, file=file)
                
if __name__ == "__main__":
    test()
    
#%%
## Best Score: 200.18515048829298
