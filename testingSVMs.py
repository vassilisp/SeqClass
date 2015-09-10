# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:32:22 2015

@author: vpan
"""


from testingfileloader import loadclassifiers

from sklearn.grid_search import GridSearchCV
from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold

from stratified_metrics import EVALUATE_TEST
import time
import Globals
from pipelineC_T_SGD import findclf_name
import LoadingTestData
import rebatcher
import numpy as np


#%%
proID = 'pro288817'
tokens = 3

exeTime = time.strftime('%d%m_%H%M')

from reporter import Reporter

genReport = Reporter()
genReport.new_report('GridSearch for SVM C value')
def tester(Xor, Yor, proID, tokens, div):
    #%%
    localreport = Reporter()
    localreport.subreport(str(proID)+'_'+str(tokens)+'_'+str(div))
    X, Y, repor = rebatcher.single_rebatcher(Xor,Yor, div)
    localreport.concat_report(repor)
    #%%
    
    cv = StratifiedKFold(Y, 4)
    
    classifier_dic = loadclassifiers()
    
    svmpipe = classifier_dic['LinearSVC ']
    svmpipe = clone(svmpipe)
    
    c = np.arange(-10,5,0.1 )
    xx = np.ones(len(c))*2
    xx = np.power(xx,c)
    xx = np.array([1,2])
    svm_params ={'clf__C': xx.tolist()
                                                                }# 0.1, 1, 10, 100),
    
    scoring = 'accuracy'
    grid3 = GridSearchCV(estimator=svmpipe, n_jobs=-1, refit=False,
                                        param_grid=svm_params, cv=cv, scoring = scoring, error_score=0,
                                        verbose = 10)
    #%%               
    t1 = time.time()
    grid3.fit(X,Y)
    t1 = time.time() - t1
    
    #55
    estim = clone(grid3.estimator)
    best_params = grid3.best_params_.copy()
    #the best estimator is the estimator of the pipe set with the best
    #parameters of the pipe
    best_estimator = estim.set_params(**best_params)
    
    clf_name = findclf_name(best_estimator)
    
    try:
        best_estimator.set_params(**{'clf__estimator__probability': True})
    except:
        pass
    
    if clf_name == 'SVC':
        try:
            clf_name += '_' + best_estimator.named_steps['clf'].kernel
        except:
            print('probably not an SVC')
    
    dr_name = 'noDR'
    pipe_desc = 'testingSVDgs'
    filename_starter = str(proID) + '_' + str(tokens) + '_' + str(div) + '_'           
    filename_descr = pipe_desc + '_' +dr_name +'_'+ clf_name
    filename_full =  filename_starter + filename_descr
    from main import reportSCORES
    from pipelineC_T_SGD import savejson
    
    savejson(best_params, path + 'best_params/', filename_full + '_bestparams.txt')
    text, top_scores = reportSCORES(grid3.grid_scores_[:20],name=clf_name, pipe_de=pipe_desc, dr = dr_name)


    localreport.report('Found best parameters')
    localreport.report('-Execution time: ' + str(t1))
    localreport.report(str(best_params))
    localreport.report('-----scores and parameters------')
    localreport.report(str(text))
    
    localreport.saveReport(path+'report/', filename_full + "_report.txt")
    genReport.concat_report(localreport)
    genReport.saveReport(path + 'report/', filename_full + "_Report.txt" )
    genReport.saveReport(path + 'report/', filename_descr + "_Report.txt" )
    
    return {'LinearSVC-gs-' + str(div):best_estimator}
    



if __name__ == '__main__':

    proID = 'pro288817'
    tokens = 3

    classifier_dic = loadclassifiers()

    #div = simple and sliding

    divers_sliding = {batchN:25, min_div:100, max_div:200}
    divers_simple = {batchN:100, min_div:0, max_div:0}
    for pages in (True, False):
        for tok in tokens:
            estimators = {}
            estimators.update({'(200)LinearSVC': classifier_dic['LinearSVC ']})
            estimators.update({'(200)MultinomialNB': classifier_dic['MultinomialNB ']})
                
            X, Y = LoadingTestData.loadTestData(proID, 'clientId',tok, onlyPages=pages)
        
            for div in (200,100,50,25):
                #grid search for optimized C parameter - can be omited
                entry = tester(X, Y, proID, tokens, div)
    
                best_estimators = estimators.copy()
                best_estimators.update(entry)


                method = 'T' + str(tok) +'-SVMGS-P('+str(pages)+')'  +proID
                
                path = Globals.getProcessIDPath(method, exeTime)
                
                #choose divider method - many options such as simple, accumulating and sliding window
                Xdiv, Ydiv, rep = rebatcher.single_rebatcher(X,Y, div)
                EVALUATE_TEST(Xdiv,Ydiv, best_estimators, path+str(div)+'/', method + '_' + str(div))        
#%%

    

