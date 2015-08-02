# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:43:50 2015

@author: vpan
"""

import Globals

import pickle
import json

from reporter import Reporter

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
from time import time, ctime

from sklearn.cross_validation import StratifiedKFold
#%%    
def getPipelines(Y_develop='', SELECT_PIPELINE = 1):
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
    estimators = get_estimators()
    
    #%%
    estimators_grid1 = {}
    estimators_grid2 = {}
    estimators_grid3 = {}    
    cv = StratifiedKFold(Y_train, 5)
    scoring = 'accuracy'
    dense = Densifier()  
    for estimator, estimation_params in estimators.items():

        print('-'*60)
        print('Start testing estimator') 
            
        if SELECT_PIPELINE == 'basic':
            CLF_pipe_basic = Pipeline([
                             ('tfidfVec', TfidfVectorizer()),
                             ('gus', GenericUnivariateSelect(mode='percentile')),
                             ('clf', estimator)])      
            #pack preprocessing params and estimation params together
            params_basic = pre_params.copy()
            #params_basic.update(estimation_params)

            grid1 = GridSearchCV(estimator=CLF_pipe_basic, n_jobs=-1,
                                param_grid=params_basic, cv=cv, scoring = scoring,
                                verbose=10, refit=False)
            estimators_grid1.update({'basic':grid1})
            

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
            estimators_grid2.update({'STD':grid2})
            
            
        if SELECT_PIPELINE == 'full':
            #SUPER EXPENSIVE SEARCH - probably better using RandomizedGridCV
            
            #DRS = (PCA(), TruncatedSVD(), RandomizedPCA(), FastICA(), NMF(), EMPTY())  

            DRS = get_DRS()                
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
                                    
                grid3 = GridSearchCV(estimator=CLF_pipeDR, n_jobs=-1, refit=False,
                                    param_grid=params_DR, cv=cv, scoring = scoring,
                                    verbose = 10)
                estimators_grid3.update({'full':grid3})
    
    
    ##Gather all generated estimator pipelines:
    generated_pipelines = {}
    generated_pipelines.update(estimators_grid1)
    generated_pipelines.update(estimators_grid2)
    generated_pipelines.update(estimators_grid3)
    #%%
    """
    gs = estimators_grid2[0]
    gs.fit(X_train, Y_train)    
    gs.best_score_
    gs.best_params_
    #gs.grid_scores_
    
    #%%
    """
def run(X_develop, Y_develop, proID, tokens, div, div_path, SELECT_PIPELINE):
    
    generated_pipelines = getPipelines(Y_develop, SELECT_PIPELINE)
    
    kept_all_best_estimators = {}
    kept_all_best_params = {}
    kept_all_best_full_params = {}
    kept_all_top_scores = {}
    
    report = Reporter()
    if len(generated_pipelines)==1:
        print("OK")
    else:
        print('Something went wrong generating the pipelines')
        exit
    
    for pipe_desc, gen_pipes in generated_pipelines.items():
        print('\n'*10)
        print('##'*40)
        print('RUNNING PIPE :', pipe_desc)
        print('##'*40)
                
        report.new_report('RUNNING PIPE: ' + pipe_desc)
        
        for cnt, estimator_pipe in enumerate(gen_pipes):        
            print('=='*40)
            print('RUNNING PIPE :', pipe_desc)
            print('__'*40)
            print(estimator_pipe)
            print('=='*40)
            
            clf_name = estimator_pipe.estimator.named_steps['clf'].__class__.__name__
            
            report.subreport('Testing estimator : ' + clf_name + ' in ' + pipe_desc + ' pipe')
                
            t0 = time()
            estimator_pipe.fit(X_train,Y_train)
            t1 = time()-t0
            

            estim = estimator_pipe.estimator            
            best_params = estimator_pipe.best_params_
            #the best estimator is the estimator of the pipe set with the best
            #parameters of the pipe
            best_estimator = estim.set_params(**best_params)
            
            steps = best_estimator.named_steps
            dr_dic = {}            
            dr_name = ''            
            if 'dr' in steps:
                dr_dic = {'dr':str(steps['dr'])}
                dr_name = str(steps['dr'])
            
            best_full_params = best_params.copy()
            best_full_params.update(dr_dic)
            best_full_params.update({'clf':clf_name})
            

            pipe_path = div_path + '/' + pipe_desc + '_pipe_details/'
            filename_starter = proID + '_' + tokens + '_' + div            
            filename_descr = pipe_desc + '_' +dr_name +'_'+ clf_name
            filename_full =  filename_starter + filename_descr
            
            #save estimator, best_params and best_full_params
            savejson(best_params, pipe_path + 'best_params/', filename_full + '_bestparams.txt')
            savejson(best_full_params, pipe_path + 'best_full_params/', filename_full + '_bestfullparams.txt')
            savepickle(best_estimator, pipe_path + 'estimators/', filename_full +'_bestestimator.pickle')            
            
            text, top_scores = report(estimator_pipe.grid_scores_[:20],name=clf_name, pipe_de=pipe_desc, dr = dr_dic)
            
            clf_descr = clf_name
            if dr_name != '':
                clf_descr += ' (' + dr_name + ')'
            
            kept_all_best_params.update({clf_descr:best_params})
            kept_all_best_full_params.update({clf_descr:best_full_params})
            kept_all_best_estimators.update({clf_descr:best_estimator})
            kept_all_top_scores.update({clf_descr: top_scores})
            
            report.report('Found best parameters')
            report.report('-Execution time: ' + t1)
            report.report(str(best_full_params))
            report.report(text)
            report.saveReport(pipe_path +'/reports/', filename_starter + '_' + pipe_desc + '_' + cnt + '_report.txt')
            
        
        filename_starter = proID + '_' + tokens + '_' + div + '_' + pipe_desc + '_pipe'     
        report.saveReport(div_path + 'reports/', filename_starter + '_report.txt')
    
        #Finished pipe execution -Evaluate results
    
        #save estimator, best_params and best_full_params
        savejson(kept_all_best_params, div_path + 'DATA/best_params/', filename_starter + '_keptbestparams.txt')
        savejson(kept_all_best_full_params, div_path + 'DATA/best_full_params/', filename_starter + '_keptbestfullparams.txt')
        savepickle(kept_all_best_estimators, div_path + 'DATA/estimators/', filename_starter +'_keptbestestimator.pickle')
        
        #evaluate all saved estimators up to that point
        #pass estimator for evaluation        
        EVALUATE_estimator(kept_all_best_estimators, div_path +'EVALUATION/', file_starter)
        EVALUATE_top_scores(kept_all_top_scores, div_path + 'EVALUATION/', file_starter)
        
    
    return kept_all_best_estimators, kept_all_top_scores
        




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
       
    
        
        
#%%
#%% DEFINE ALL RUNNING PARAMETERS HERE [get_pre_params], [get_std_params], [get_estimators], [get_DRS]
#%% 
#%%Define Preprocessing and feature extractions steps      
def get_pre_params():

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
    return pre_params

    
#%%
def get_std_params():
    
    std_params = {}
    #stad_params = {'stad__with_mean': [True, False],
    #                   'stad__with_std': [True, False]}
    return std_params
    

#%%Define classifier parameters                
def get_estimators():
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

    #Pack estimators into dictionary
    
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
    return estimators


#%%Dimensionality reduction parameters
def get_DRS():
    empty = EMPTY()
    return (emtpy, TruncatedSVD(), PCA())
def get_dr_params():
    dr_params = {}
    dr_params = {'dr__n_component': (100, 1500)# 'mle',100,500,1000,3000
                }
    return dr_params
    
#%%
# Utility function to report best scores
def report(grid_scores,name='', pipe_de= '', dr='', n_top=5):
    a = '\n' + 'Report for best estimator '+ name + ' in' + pipe_desc 

    a = 'Best estimator uses DR : ' + str(dr) + '\n'
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        
        mean_validation_score = score.mean_validation_score
        std_mean_validation_score = np.std(score.cv_validation_scores)
        a = "Mean validation score: {0:.3f} (std: {1:.3f})".format(
              mean_validation_score, std_mean_validation_score)
        a += "Parameters: {0}".format(score.parameters)
        a +='\n'
    return a, top_scores
        
        
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
        if 'n_components' in kargs:
            if kargs['n_components'] != 100:
                raise FitFailedWarning
        return {'EMPTY': 'Panopoulos ;)'}

        
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
    X_train,Y_train = LoadingTestData.loadTestData('pro307653', 'clientId', 3)
    #%%    
    run()