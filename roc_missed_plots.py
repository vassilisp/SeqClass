# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 23:50:43 2015

@author: vpan
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
import matplotlib.pyplot as plt

def process_ALL(X_validation, Y_validation, estimators, path, method):
    
    lb = LabelBinarizer()
    y_real = lb.fit_transform(Y)
    
    
    
    y_df_all = {}
    y_pb_all = {}    
    for descr, estimator in estimators_descr.items():
        if hasattr(estimator, 'decision_function'):
            y_df = estimator.decision_function            
            y_df_all.update(descr, y_df)
        if hasattr(estimator, 'predict_proba'):
            y_pb = estimator.predict_proba
            y_pb_all.update(descr, y_pb)
    
    y_graph = {'DecF':y_df, 'PredP':y_pb}
    
    
    for func_descr, func in y_graph.items():
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        fig1.hold(True)
        fig2.hold(True)
        fig3.hold(True)    
        for descr, actual_function in func.items():    
            try:
                y_score = actual_function(X)
                fpr, tpr = metrics.roc_curve(y_real.ravel(), y_score.ravel())
                
                auc = metrics.auc(fpr, tpr)
    
                tpr[0] = 0.0
                tpr[-1] = 1.0
                
                fig1.plot([0,1],[0,1], 'g--', lw=1)    
                fig1.plot(fpr, tpr,'b-', label=descr + ' :area=%0.2f)'%auc, lw=2)
                
    
                misal = 1-tpr
                tpr[1] = 1.0
                tpr[-1] = 0.0    
                fig2.plot(fpr, misal,'b-', label= descr + ' :area=%0.2f)'%auc, lw=2)
                fig3.plot(fpr, misal,'b-', label= descr + ' :area=%0.2f)'%auc, lw=2)
                
            except:
                pass            
        
        fig1.xlabel('False Positive (%)')
        fig1.ylabel('True Positive (%)')
        fig1.grid(which='major')
        fig1.grid(True, which='minor')
        fig1.legend(loc='best')    
        
        fig2.xlabel('False alarms (%)')
        fig2.ylabel('Missed alarms (%)')
        fig2.xscale('log')
        fig2.xlim(0.001, 1)
        fig2.grid(which='major')
        fig2.grid(True, which='minor')
        fig2.legend(loc='best')
        
        fig3.xlabel('False alarms (%)')
        fig3.ylabel('Missed alarms (%)')
        fig3.grid(which='major')
        fig3.grid(True, which='minor')
        fig3.legend(loc='best')    
        
        fdescr = proID +'_' + div +'_'+ func_descr + '_
        savefig(fig1, path,  fdescr + 'roc_curve.svg')
        savefig(fig2, path,  fdescr + 'missed_alarms_log.svg')
        savefig(fig3, path,  fdescr + 'missed_alarms.svg')
#%%
            
        
    
    
def SCORER_GRAPHER(X, Y, estimator, path, method):
    
    estimator1 = SVC()
    estimator2 = MultinomialNB()
    
    lb = LabelBinarizer()
    y_real = lb.fit_transform(Y)
    y_pred = estimator1.predict(X)
    
    if hasattr(estimator1, 'predict_proba'):
        try:        
            y_proba = estimator1.pred_proba(X)
        except:
            y_proba = print("Unexpected error:", sys.exc_info()[1])
    
    if hasattr(estimator1, 'decision_function'):
        try:
            y_decfun = estimator1.decision_function(X)
        except:
            y_decfun = y_proba = print("Unexpected error:", sys.exc_info()[1])
            
            
    fpr, tpr, thr = metrics.roc_curve(y_real.ravel(), y_proba.ravel())
    

    #%% GENERATE and save ROC and MISSED
    fdescr = proID +'_' + div +'_'+ method
    
    auc = metrics.auc(fpr, tpr)
    plt.hold(True)
    plt.plot([0.01,1],[0,1], 'g--', lw=1)    
    plt.plot(fpr, tpr,'b-', label='AUC: %0.2f)'%auc, lw=2)
    tpr[0] = 0.0
    tpr[-1] = 1.0
    plt.title('ROC curve-' + str(method))
    plt.xlabel('false positive/false alarm (%)')
    plt.ylabel('True Positive (%)')
    plt.grid(which='major')
    plt.grid(True, which='minor')
    plt.legend(loc='best')        
    savefig(plt, path, fdescr + 'roc_curve.svg' )

    
    plt.hold(False)
    misal = 1-tpr
    plt.plot(fpr, misal,'b-', label='AUC: %0.2f)'%auc, lw=2)
    tpr[1] = 1.0
    tpr[-1] = 0.0
    plt.title()
    plt.xlabel('False alarms (%)')
    plt.ylabel('Missed alarms (%)')
    plt.xscale('log')
    plt.xlim(0.001, 1)
    plt.grid(which='major')
    plt.grid(True, which='minor')
    plt.legend(loc='best')
        
    savefig(plt, path,  fdescr + 'missed_alarms.svg')
    
    

    
def savefig(plt, path, filename):
    plt.savefig(path + filename, dpi=1200, format='svg' )