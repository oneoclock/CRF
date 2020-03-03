# -*- coding: utf-8 -*-

#%%
import numpy as np
class CRFModel():
    """
    CRF Model wrapper with utils to convert labels to index and handle
    column-major order
    """
    def __init__(self, dimX=128, dimY=26):
        """
        dimX: The number of features of each word
        dimY: The number of labels
        """
        self.dimX = dimX
        self.dimY = dimY
        self.labels = tuple(range(1, self.dimY+1))
        
    def load_WT(self, W, T):
        """
        Load W and T
        W is expected to be DIMX x DIMY
        T is transposed DIMY x DIMY
        """
        self._W = W
        self._T = T
        
    def load_X(self, x, from_file=False):
        if from_file:
            W, T = self._load_wt_file(x)
        else:
            W, T = self._load_wt(x)
        self._W = W
        self._T = T
            
        
    def _load_wt(self, x):
        """
        Loading sequence of W,T from python object.
        Reflecting how we get from file.
        The convention is :
            W is expected to be 128 x n_labels
            T is expected to be 26 x 26 but transposed
        """
        W = x[:self.dimY*self.dimX].reshape(self.dimY,self.dimX) 
        W = W.transpose() # Coz W is stored in such way
    #    print(W)
        T = x[self.dimY*self.dimX:].reshape(self.dimY,self.dimY).transpose() 
        return W, T

        
    def _load_wt_file(self, x):
        """
        Loading sequence of W,T from file. Either x is a file path or sequence
        The convention is :
            W is expected to be 128 x n_labels
            T is obtianed in transposed order
        """            
        if type(x) == str:
            # Load file
            x = np.loadtxt(x)
            
        W = x[:self.dimY*self.dimX].reshape(self.dimY,self.dimX)
        W = W.transpose() # convention
        T = x[self.dimY*self.dimX:].reshape(self.dimY,self.dimY) # no need to transpose
        return W, T
        
    def getT(self, label_current,label_next):
        i = np.array(label_current, dtype='int32')
        j = np.array(label_next, dtype='int32')
        #handling column major order
        return self._T[j-1, i-1]
        
    def getW(self, label):
        # backward compatible
        return self._W[:,np.array(label, dtype='int32') - 1]