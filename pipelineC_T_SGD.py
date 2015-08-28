# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:43:50 2015

@author: vpan
"""

import Globals

import pickle
import json

from reporter import Reporter

import numpy as np

from operator import itemgetter

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsRestClassifier
#import feature extraction and Preprocessing modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer

#import feature select modules
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2, f_classif

#import dimensionality redaction modules
from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA, NMF, FactorAnalysis, TruncatedSVD, FastICA

#import Pipeline and Search Modules
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from time import time, ctime

from sklearn.cross_validation import StratifiedKFold

from sklearn.base import clone
from stratified_metrics import EVALUATE_TEST, EVALUATE_TOPSCORES
#%%    
def getPipelines(Y, SELECT_PIPELINE = 'basic'):
    """
    Path = 'proId / tokens / div'
    Select which pipeline to run 
    //POSIBLE OPTIONS   [1] for basic_pipe
                        [2] for pipe with standardization step
                        [3] for full pipe with Dimensionality reduction
                        
    WARNING: 2 is same as 1 when run with STD:FALSE,FALSE and much slower
    """
    
    #%%TEST AWAY
    pre_params = get_pre_params()
    std_params = get_std_params()
    dr_params = get_dr_params()
    estimators, estimator_params = get_estimators()
    
    #%%
    estimators_grid1 = []
    estimators_grid2 = []
    estimators_grid3 = []    
    cv = StratifiedKFold(Y, 3)
    scoring = 'accuracy'
    dense = Densifier()  
    for estimator, estimation_params in zip(estimators, estimator_params):

        print('-'*60)
        print('Start MAKING some PIPES') 
            
        if SELECT_PIPELINE == 'basic' or SELECT_PIPELINE == 1:
            CLF_pipe_basic = Pipeline([
                             ('tfidfVec', TfidfVectorizer()),
                             ('gus', GenericUnivariateSelect(mode='percentile')),
                             ('clf', estimator)])      
            #pack preprocessing params and estimation params together
            params_basic = pre_params.copy()
            params_basic.update(estimation_params)

            grid1 = GridSearchCV(estimator=CLF_pipe_basic, n_jobs=-1,
                                param_grid=params_basic, cv=cv, scoring = scoring,
                                verbose=10, refit=False)
            estimators_grid1.append(grid1)
            

        if SELECT_PIPELINE == 'STD':
            CLF_pipeSTD = Pipeline([
                             ('tfidfVec', TfidfVectorizer()),
                             ('gus', GenericUnivariateSelect(mode='percentile')),
                             ('toDense', dense),
                             ('stad', StandardScaler()),
                             ('clf', estimator)])      
            #pack preprocessing params and estimation params together
            params_STD = pre_params.copy()
            params_STD.update(std_params)
            params_STD.update(estimation_params)

            grid2 = GridSearchCV(estimator=CLF_pipeSTD, n_jobs=-1, refit=False,
                                param_grid=params_STD, cv=cv, scoring = scoring,
                                verbose = 10)
            estimators_grid2.append(grid2)
            
            
        if SELECT_PIPELINE == 'full'or SELECT_PIPELINE == 3:
            #SUPER EXPENSIVE SEARCH - probably better using RandomizedGridCV
            
            #DRS = (PCA(), TruncatedSVD(), RandomizedPCA(), FastICA(), NMF(), noDR())  

            DRS = get_DRS()                
            for DR in DRS:
                CLF_pipeDR = Pipeline([
                                 ('tfidfVec', TfidfVectorizer(ngram_range=((1,1)),binary=True, sublinear_tf=True)),
                                 ('gus', GenericUnivariateSelect(mode='percentile')),
                                 ('toDense', dense),
                                 #('dr', DR),
                                 #('stad', Normalizer()),
                                 ('clf', estimator)])
                #pack preprocessing params and estimation params together
                params_DR = pre_params.copy()
                params_DR.update(std_params)
                params_DR.update(dr_params)
                params_DR.update(estimation_params)            
                                    
                grid3 = GridSearchCV(estimator=CLF_pipeDR, n_jobs=-1, refit=False,
                                    param_grid=params_DR, cv=cv, scoring = scoring, error_score=0,
                                    verbose = 10)
                estimators_grid3.append(grid3)

    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #Custom extra pipes    
    CLF_pipeEXTRA = Pipeline([
                             ('tfidfVec', TfidfVectorizer(ngram_range=((1,1)),binary=True, use_idf=True, sublinear_tf=True)),
                             ('gus', GenericUnivariateSelect(mode='percentile')),                             
                             ('clf', MultinomialNB())])
    
    params_custom = pre_params.copy()
    params_custom.update({'clf__alpha': [0.1, 0.001],
                          'tfidfVec__ngram_range':[(3,3),(4,4)]})
        
    grid4 =GridSearchCV(estimator=CLF_pipeEXTRA, n_jobs=-1, refit=False,
                                    param_grid=params_custom, cv=cv, scoring = scoring, error_score=0,
                                    verbose = 10)
    #INCLUDE CUSTOM NB OR NOT
    estimators_grid3.append(grid4)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
    
    ##Gather all generated estimator pipelines:
    generated_pipelines = {}
    generated_pipelines.update({'basic':estimators_grid1})
    generated_pipelines.update({'withSTD':estimators_grid2})
    generated_pipelines.update({'basic':estimators_grid3})
    
                           
                             
    
    return generated_pipelines
    #%%
    """
    gs = estimators_grid2[0]
    gs.fit(X_train, Y_train)    
    gs.best_score_
    gs.best_params_
    #gs.grid_scores_
    
    #%%
    """

        




def savepickle(data, path, filename):
    #make path first
    Globals.mkdir_LR(path)
    with open(path+filename, 'wb') as outfile:
        pickle.dump(data, outfile)
def savejson(data, path, filename):
    #make path first
    Globals.mkdir_LR(path)
    with open(path+filename, 'w') as outfile:
        json.dump(data, outfile)
       
    
def findclf_name(estimator):

    testim = clone(estimator)
    try:
        while (not isinstance(testim, BaseEstimator)):
            testim = testim.estimator
    except:
        pass
    
    clf_name = testim.__class__.__name__
    if clf_name == 'Pipeline':
        clf_name = testim.named_steps['clf'].__class__.__name__
        testim = testim.named_steps['clf']
    
    if clf_name == 'OneVsRestClassifier':
        clf_name = testim.estimator.__class__.__name__
        try:
            clf_name += '_' + testim.estimator.kernel
        except:
            pass
    
    return clf_name
    
        
#%%
#%% DEFINE ALL RUNNING PARAMETERS HERE [get_pre_params], [get_std_params], [get_estimators], [get_DRS]
#%% 
#%%Define Preprocessing and feature extractions steps      
def get_pre_params():

    tfidfVec_params = {'tfidfVec__ngram_range': ((1,1),(2,2)),#(3,3),(4,4)),#,(4,4)),#,(8,8)),#, (1,1)), #(2,3),(3,3),(3,4),(4,4)),#,(5,5)),#(13,15),(18,20), (10,12), (15,15)),
                        #'tfidfVec__max_df': (0.9, 0.7),# 0.5, 0.3),
                        #'tfidfVec__binary': (True, False),
                        #'tfidfVec__norm': (None, 'l1', 'l2'),
                        #'tfidfVec__use_idf': (True, False),
                        #'tfidfVec__sublinear_tf': (True, False)
                        }
    gus_params = {'gus__param': (15, 40, 65)}
    #gus_params = {'gus__param': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    #                  'gus_score_func': (f_classif(), chi2())}
    
    ##Pack preprocessing params together and ready to go
    pre_params = {}
    pre_params.update(tfidfVec_params)
    pre_params.update(gus_params)
    return pre_params

    
#%%
def get_std_params():
    
    std_params = {}
    #stad_params = {'stad__with_mean': [True, False],
    #                   'stad__with_std': [True, False]}
    return std_params
    

#%%Define classifier parameters                
def get_estimators():
    lSVC_params = {
                    #'clf__estimator__C': [0.001, 0.01, 1, 10],# 0.1, 1, 10, 100),
                      #'clf__gamma': (1, 0.1)}# 0.01)#, 0.001, 0.0001, 0.00001)
                   }
    
    SVC_params = {#'clf__estimator__C': (0.001, 0.01, 1, 10),# 1, 10, 100),
                  #'clf__gamma': (1, 0.01, 0.0001,)# 0.0001, 0.00001)}                  
                  #'clf__estimator__kernel': ('rbf', 'poly', 'sigmoid')
                  }
    
    SGD_params = {#'clf__alpha': (10.0 ,1.0 ,0.1, 0.001, 0.0001, 0.000001)
                    }
    
    MNB_params = {#'clf__alpha': [1, 0.1]#, 0.001, 0.0001, 10)}
                  }
    
    DT_params = {#'clf__criterion': ('gini', 'entropy'),
                 #'clf__min_samples_split': (2,4,8,16,32,64,100),
                 #'clf__min_samples_leaf': (1,5,10,15,20)
                 }
                    
    RFC_params = {}
    
    ABC_params = {}
    
    LDA_params = {}
    
    QDA_params = {}

    #Pack estimators into dictionary
    
    #ovrSVCrbf = OneVsRestClassifier(SVC(kernel='rbf'))
    #ovr_lsvc = OneVsRestClassifier(SVC(kernel='linear'))
    ovr_lsvc = LinearSVC()
    
    estimators = []
    estimators_params = []

    #estimators.append(LDA())
    #estimators_params.append(LDA_params)

    #estimators.append(QDA())
    #estimators_params.append(QDA_params) 

    #estimators.append(MultinomialNB())
    #estimators_params.append(MNB_params)
    
    estimators.append(ovr_lsvc)
    estimators_params.append(lSVC_params)

    #estimators.append(ovrSVCrbf)
    #estimators_params.append(SVC_params)

    #estimators.append(DecisionTreeClassifier())
    #estimators_params.append(DT_params)

    
    #sgd = SGDClassifier(n_jobs=-1)
    #estimators.append(sgd)
    #estimators_params.append(SGD_params)

       
    
    """
    estimators = {ovr_lsvc: lSVC_params, 
                  MultinomialNB(): MNB_params,
                  ovrSVCrbf: SVC_params,
                  DecisionTreeClassifier(): DT_params,
    #              RandomForestClassifier(): RFC_params,
                  AdaBoostClassifier(): ABC_params,
                  LDA(): LDA_params,
    #              QDA(): QDA_params
                  }
    """
    return estimators, estimators_params


#%%Dimensionality reduction parameters
def get_DRS():
    empty = noDR()
    res = []
    
    #res.append(empty)
    ##res.append(FastICA()) -- EXTREMELY SLOW, DONT TRY IT OUT
    res.append(TruncatedSVD())
    #res.append(RandomizedPCA())
    return res
def get_dr_params():
    dr_params = {}
    dr_params = {#'dr__n_components': (10, 50,100, 500)#,500, 1500)# 'mle',100,500,1000,3000
                }
    return dr_params
    
#%%
# Utility function to report best scores
def reportSCORES(grid_scores,name='', pipe_de= '', dr='', n_top=5):
    a = '\n' + 'Report for best estimator '+ name + ' in ' + pipe_de

    a += ' ,Best estimator uses DR : ' + str(dr) + '\n'
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top+1]
    for i, score in enumerate(top_scores):
        
        mean_validation_score = score.mean_validation_score
        std_mean_validation_score = np.std(score.cv_validation_scores)
        a += "Mean validation score: {0:.3f} (std: {1:.3f})".format(
              mean_validation_score, std_mean_validation_score)
        a += "Parameters: {0}".format(score.parameters)
        a +='\n'
    
    return a, top_scores
        
        
#%%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import FitFailedWarning


class noDR(BaseEstimator, TransformerMixin):
    ncomp = 0
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y,**fit_params)        
        return self.transform(X)
    def transform(self, X, y=None, **fit_params):
        if self.ncomp == 100:
            return X
        else:
            #raise FitFailedWarning
            return X
        
    def set_params(self, **kargs):
        if 'n_components' in kargs:
            self.ncomp = kargs['n_components']
        return {'noDR': 'Panopoulos ;)', 'n_components':self.ncomp}
    
    

        
#%%
class Densifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y,**fit_params)        
        return self.transform(X)
    def transform(self, X, y=None, **fit_params):
        return X.todense()

        
#%%        
if __name__ == "__main__":
    #%%dummy data
    import LoadingTestData

    #X_train,Y_train = LoadingTestData.loadTestData('pro307653', 'clientId', 3)
    #%%    
    from sklearn.datasets import make_classification, fetch_20newsgroups
    X,Y = make_classification(n_samples=200, n_classes=4, n_features=20, n_informative=10)
    
    

    categories = ['alt.atheism', 'talk.religion.misc',
               'comp.graphics', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)    
    Y = newsgroups_train.target[:20]
    X = newsgroups_train.data[:20]
    
    
    run(X,Y, 'test', '4', 'full', '/home/vpan/TESTPIPE/', 1 )
