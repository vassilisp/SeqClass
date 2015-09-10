# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:23:08 2015

@author: vpan
"""

from sklearn.cross_validation import _BaseKFold
from sklearn.cross_validation import StratifiedKFold
import rebatcher
import LoadingTestData
import numpy as np

class MasquerFold(_BaseKFold):
    _indices = None
    #y = None
    def __init__(self, y,inter, n_folds=3, shuffle=False, random_state=None):
        self.y = y
        
        self.shuffle = shuffle
        self.random_state=random_state
        self.n=len(inter)
        self.n_folds = n_folds
        self.inter = inter
        
    def _iter_test_indices(self):
        skf = StratifiedKFold(self.y, self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        it = iter(skf)
        for cn in range(self.n_folds):   
            print(cn)
            index = next(it)
            #tmp2 = []
            tmp = np.zeros(self.inter.shape, bool)
            for i in index[1]:
                #tmp = np.where(self.inter==i)
                oring = self.inter == i
                tmp = np.logical_or(tmp,oring)
                
                #tmp2.append(np.where(self.inter == i))
                
            yield tmp
        
        
    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i, shuffle=%s, random_state=%s)' % (
        self.__class__.__module__,
        self.__class__.__name__,
        self.n,
        self.n_folds,
        self.shuffle,
        self.random_state,
        )
        
    def __len__(self):
        return self.n_folds
        
        
if __name__ == '__main__':
    proID = 'pro48937'
    tokens = 2
    X, Y = LoadingTestData.loadTestData(proID, 'clientId',tokens)
    
    Xdiv5, Ydiv5, inter1, rep = rebatcher.single_rebatcher2(X,Y, 25, acc=True, min_div=150, max_div=300, get_inter=True)
    
    
    
    mcv = MasquerFold(Y, inter1, random_state=None)
    
    for train,test in mcv:
        print(train, '='*10, train.shape, test, '='*20, test.shape)
    

    for train,test in mcv:
        print(train, '='*10, train.shape, test, '='*20, test.shape)    
    from sklearn.cross_validation import check_cv
    
    cv = check_cv(mcv, Xdiv5, Ydiv5)
    
    print(cv)