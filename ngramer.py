# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:54:58 2015

@author: vpan
"""


#get users

#get clientIds for each user

#get transitions for each client ID and start


import matplotlib.pyplot as plt

import LoadingTestData
from sklearn.feature_extraction.text import TfidfVectorizer


def ngramer(proID):
    #proID = 'pro208105'
    
    plt.figure(figsize=(9,6))
    plt.hold(True)

    #%%
    tokens = ['0','1','2','3','4']
    for token in tokens:
        print('Running for token:', token)
        
        X, Y = LoadingTestData.loadTestData(proID, 'clientId',token)
    
        
    #%%
        results = []
        for i in range(1,7):
            tf = TfidfVectorizer(ngram_range=(i,i))
            tf.fit(X)
            print('--calculated ngram', i)
            results.append(len(tf.get_feature_names()))
        
        #%%
        
        print('----Ploting result and proceeding')
        plt.plot(range(1,10), results, label='URL ' + token)
        plt.legend(loc='best')
        plt.ylabel('Features')
        plt.xlabel('ngram extraction')
        plt.title('Resulting features for different ngrams (' + proID + ')')
    
    
    plt.savefig('ngram_' + proID, dpi=600, format='png')    
    
    
if __name__ == '__main__':
    proIDS = ['pro208959', 'pro208105']
    
    for pro in proIDS:
        ngramer(pro)
        
    