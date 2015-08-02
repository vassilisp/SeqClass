# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:43:50 2015

@author: vpan
"""

#import classifiers
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.multiclass import OneVsRestClassifier
#import feature extraction and Preprocessing modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

#import feature select modules
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2, f_classif

#import dimensionality redaction modules
from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA, NMF, FactorAnalysis, TruncatedSVD, FastICA

#import Pipeline and Search Modules
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import StratifiedKFold
#%%
def run():
    clf_linker = {}
    param_linker = {}

    #%%Define Preprocessing and feature extractions steps
    tfidfVec_params = {'tfidfVec__ngram_range': ((1,1),(2,2))#,(2,3)),#,(3,3),(3,4),(4,4),(13,15),(18,20), (10,12), (15,15)),
                        #'tfidfVec__max_df': (0.9, 0.7),# 0.5, 0.3),
                        #'tfidfVec__binary': (True, False),
                        #'tfidfVec__norm': (None, 'l1', 'l2'),
                        #'tfidfVec__use_idf': (True, False),
                        #'tfidfVec__sublinear_tf': (True, False)}                    
                        }
    gus_params = {'gus__param': (30, 60)}
    #gus_params = {'gus__param': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    #                  'gus_score_func': (f_classif(), chi2())}
    
    ##Pack preprocessing params together and ready to go
    pre_params = {}
    pre_params.update(tfidfVec_params)
    pre_params.update(gus_params)
    
    #%%
    std_params = {}
    #stad_params = {'stad__with_mean': [True, False],
    #                   'stad__with_std': [True, False]}
    
    #%%Define classifier parameters                
    lSVC_params = {'clf__C': [0.001,]# 0.01],# 0.1, 1, 10, 100),
                      #'clf__gamma': (1, 0.1)}# 0.01)#, 0.001, 0.0001, 0.00001)
                   }
    
    SVC_params = {'clf__C': (0.001, 0.01, 0.1),# 1, 10, 100),
                  'clf__gamma': (1, 0.1, 0.01),#, 0.001, 0.0001, 0.00001)}                  
                  'clf__kernel': ('rbf', 'poly', 'sigmoid')
                  }
    
    SGD_params = {'clf__alpha': (0.1, 0.001, 0.0001, 0.000001)}
    
    MNB_params = {'clf__alpha': [1, 0.1]#, 0.001, 0.0001, 10)}
                  }
    
    DT_params = {'clf__criterion': ('gini', 'entropy'),
                 'clf__min_samples_split': (2,4,8,16,32,64,100),
                 'clf__min_samples_leaf': (1,5,10,15,20)
                 }
                    
    RFC_params = {}
    
    ABC_params = {}
    
    LDA_params = {}
    
    QDA_params = {}
    
    #%%Dimensionality reduction parameters
    dr_params = {}
    dr_params = {'dr__n_component': (100, 1500)# 'mle',100,500,1000,3000
                }
    
    
    
    #%%Pack estimators into dictionary
    
    ovrSVC = OneVsRestClassifier(SVC())
    clf = MultinomialNB()
    estimators = {LinearSVC(): lSVC_params, 
    #              clf: MNB_params,
    #              ovrSVC: SVC_params,
    #              SVC(): SVC_params,
    #              DecisionTreeClassifier: DT_params,
    #              RandomForestClassifier: RFC_params,
    #              AdaBoostClassifier: ABC_params,
    #              LDA: LDA_params,
    #              QDA: QDA_params
                  }
    
    #%%TEST AWAY
    
    
    #%%dummy data
    import LoadingTestData
    X_train,Y_train = LoadingTestData.loadTestData('pro307653', 'clientId', 3)
    #%%
    ##Select which estimators to run 
    ##//POSIBLE OPTIONS [1],[2],[3],[1,3],[2,3] , WARNING: dont do [1,2], 2 does 1 with STD:FALSE,FALSE
    
    SELECT_PIPELINE = [1]
    estimators_grid1 = {}
    estimators_grid2 = {}
    estimators_grid3 = {}    
    cv = StratifiedKFold(Y_train, 5)
    scoring = 'accuracy'    
    for estimator, estimation_params in estimators.items():

            print('-'*60)
            print('Start testing estimator') 
            
        if SELECT_PIPELINE == 1:            
            CLF_pipe_basic = Pipeline([
                             ('tfidfVec', TfidfVectorizer()),
                             ('gus', GenericUnivariateSelect(mode='percentile')),
                             ('clf', estimator)])      
            #pack preprocessing params and estimation params together
            params_basic = pre_params.copy()
            params_basic.update(estimation_params)

            grid1 = GridSearchCV(estimator=CLF_pipe_basic, n_jobs=-1, refit=True,
                                param_grid=params_basic, cv=cv, scoring = scoring,
                                verbose = 10)
            estimators_grid1.update({'basic_pipe':grid1})
            

        if SELECT_PIPELINE == 2:
            CLF_pipeSTD = Pipeline([
                             ('tfidfVec', TfidfVectorizer()),
                             ('gus', GenericUnivariateSelect(mode='percentile')),
                             ('toDense', dense),
                             ('stad', StandardScaler()),
                             ('clf', estimator)])      
            #pack preprocessing params and estimation params together
            params_std = pre_params.copy()
            params_std.update(estimation_params)

            grid2 = GridSearchCV(estimator=CLF_pipeSTD, n_jobs=-1, refit=True,
                                param_grid=paramsSTD, cv=cv, scoring = scoring,
                                verbose = 10)
            estimators_grid2.update({'withSTD_pipe':grid2})
            
            
        if SELECT_PIPELINE == 3:
            #SUPER EXPENSIVE SEARCH - probably better using RandomizedGridCV
            
            #DRS = (PCA(), TruncatedSVD(), RandomizedPCA(), FastICA(), NMF(), EMPTY())
            dense = Densifier()    
            empty = EMPTY() 
            DRS = (empty, TruncatedSVD())                
            for DR in DRS:
                CLF_pipeDR = Pipeline([
                                 ('tfidfVec', TfidfVectorizer()),
                                 ('toDense', dense),
                                 ('stad', StandardScaler()),
                                 ('gus', GenericUnivariateSelect(mode='percentile')),
                                 ('dr', DR),
                                 ('clf', estimator)])
                #pack preprocessing params and estimation params together
                params_DR = pre_params.copy()
                params_DR.update(std_params)
                params_DR.update(dr_params)
                params_DR.update(estimation_params)            
                                    
                grid3 = GridSearchCV(estimator=CLF_pipeDR, n_jobs=-1, refit=True,
                                    param_grid=params_DR, cv=cv, scoring = scoring,
                                    verbose = 10)
                estimators_grid3.update({'withDR_pipe':grid3})
     
     ##Gather all generated estimator pipelines:
     generated_pipelines = {}
     generated_pipelines.update(estimators_grid1)
     generated_pipelines.update(estimators_grid2)
     generated_pipelines.update(estimators_grid2)
    #%%
    """
    gs = estimators_grid2[0]
    gs.fit(X_train, Y_train)    
    gs.best_score_
    gs.best_params_
    #gs.grid_scores_
    
    #%%
    """
    
        
    kept_estimators = []
    for pipe_desc, gen_pipe in generated_pipelines.items():
        for estimator in gen_pipe:        
            print('=='*40)
            print('RUNNING PIPE :', pipe_desc)
            print('__'*40)
            print(estimator)
            print('=='*40)
            estimator.fit(X_train,Y_train)
            
            kept_actual_estimators.append(estimator.best_estimator_)
            
            kept_estimators.append(estimator.best_params_.update({estimator.best_estimator_.steps[4]}))
            
            report(estimator.grid_scores_, estimator= estimator.best_estimator_
            
            #ROC of estimator and every other evaluation
                        
        
        
    
        
        
        

#%%


# Utility function to report best scores
def report(grid_scores, n_top=3, estimator):
    
    a = 'Best estimator uses DR : ' + str(estimator.steps[4][1]) + '\n'
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        
        mean_validation_score = score.mean_validation_score
        std_mean_validation_score = np.std(score.cv_validation_scores))
        a = "Mean validation score: {0:.3f} (std: {1:.3f})".format(
              mean_validation_score, std_mean_validation_score
        a += "Parameters: {0}".format(score.parameters)) +'\n'
        



#%%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import FitFailedWarning
from sklearn.decomposition import PCA


class EMPTY(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y,**fit_params)        
        return self.transform(X)
    def transform(self, X, y=None, **fit_params):
        return X
        
    def set_params(self, **kargs):
        return {'EMPTY': 'Panopoulos ;)'}
        

#%%
"""
class DR_methods(BaseEstimator, TransformerMixin):
    Select one of many Dimensionality reduction methods
    
    DR = "pass"
    __name__ = "None"
    n_components = 100
    raiseWarn = False
    def __init__(self, **kargs):
        self.DR = "pass"
        
    def set_params(self, **kargs):
        for name, value in kargs.items():
            if name=='methodbill':
                if value == 'pass':
                    self.DR = "pass"
                    #return self
                else:
                    self.DR = value
                    self.__name__ = self.DR.__class__.__name__
                            
        
        kargs = dict(**kargs)
        if 'methodbill' in kargs:        
            del kargs['methodbill']
        if 'n_components' in kargs:
            if kargs['n_components'] != 100 and self.DR == 'pass':
                raise FitFailedWarning
            self.n_components = kargs['n_components']
            if (not isinstance(self.DR, PCA)) and (kargs['n_components'] == 'mle'):
                self.DR = "pass"
                print('XXXX'*10000)
                raise FitFailedWarning
                
                
        if self.DR != "pass":
            self.DR.set_params(**kargs)                
    
    def get_params(self, **kargs):
        if self.DR != "pass":
            return self.DR.get_params(**kargs)
        else:
            return {'DECOMP': False}
                
    def fit(self, X, y=None):
        if self.DR == 'pass' and self.n_components != 100:
            print('xxxxx'*10000)
            raise FitFailedWarning
        if self.DR != 'pass':
            self.DR.fit(X,y)
            
    def transform(self, X):
        if self.DR != 'pass':
            return self.DR.transform(X)
        else:
            return X
        
    def fit_transform(self,X, y=None):
        if self.DR != 'pass':
            return self.DR.fit_transform(X,y)
        else:
            return X
            
    def __name__(self):
        if self.DR == None:
            return 'DR(None :)'
        else:
            return self.DR
    
   """ 
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
    run()