# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 23:04:35 2015

@author: vpan
"""

import pickle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from pipelineC_T_SGD import Densifier



def createLSVC():
    estimator = LinearSVC()

    
    dense = Densifier()
    single_pipe = Pipeline([
                     ('tfidfVec', TfidfVectorizer(ngram_range=((3,3)),binary=False, sublinear_tf=True, use_idf=True)),
                     #('gus', GenericUnivariateSelect(mode='percentile', param=65)),
                     ('toDense', dense),
                     ('scale', MinMaxScaler() ),
                     #('dr', DR),
                     #('stad', Normalizer()),
                     ('clf', estimator)])
    
    return single_pipe


def createNB():
    #estimator = LinearSVC()
    estimator = MultinomialNB(alpha=0.001)
    
    dense = Densifier()
    single_pipe = Pipeline([
                     ('tfidfVec', TfidfVectorizer(ngram_range=((3,3)),binary=False, use_idf=True)),
                     #('gus', GenericUnivariateSelect(mode='percentile', param=65)),
                     #('toDense', dense),
                     #('scale', MinMaxScaler() ),
                     #('dr', DR),
                     #('stad', Normalizer()),
                     ('clf', estimator)])
    
    return single_pipe



def testPipe(pipe, X, Y):    
    import sys
    import traceback
    from sklearn.base import clone
    
    estim = clone(pipe)
    
    try:
        estim.fit(X[:70], Y[:70])
        estim.predict(X[:30])
        return 'OK'
    except Exception as e:
            print(sys.exc_info())
            print(e)
            print('-'*50)
            print(traceback.print_stack())
            print('-'*50)
            print(traceback.print_exc())
            print("PIPE TEST FAILED")        
            print(pipe)
            exit
        

def savepipeline(single_pipe, filename):
    import Globals
    import os

    dirname = os.path.dirname
    
    folder = "SinglePipes"
    directory = os.path.join(dirname(dirname(__file__)), folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path = directory + '/'    
    #make path first
    
    Globals.mkdir_LR(path)
    filename += '.pickle'
    with open(path+filename, 'wb') as outfile:
        pickle.dump(single_pipe, outfile)
        
        
if __name__ == '__main__':
    import LoadingTestData     
    X,Y = LoadingTestData.loadTestData('pro208105', 'clientId', tokens=2)    
    X = X[:100]
    Y = Y[:100]    
    
    custom_pipe = createLSVC()
    testPipe(custom_pipe, X, Y)
    #savepipeline(custom_pipe, 'pro208959_2_200_noDR_noGUS_LinearSVC-s_customestimator')
    
    custom_pipe = createNB()
    testPipe(custom_pipe, X, Y)
    savepipeline(custom_pipe, 'pro208959_2_200_noDR_noGUS_MultinomialNB_customestimator')
    