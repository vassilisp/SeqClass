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
from time import time
from sklearn import metrics
import numpy as np

def EVALUATE_estimators(X_develop, Y_develop, X_validation, Y_validation, estimators, path, method):
    train_times = []
    estimation_times = []
    labels = []
    for est_descr, estimator in estimators.items():
        t0 = time()
        estimator.fit(X_develop, Y_develop)
        t1 = time() - t0        
        train_times.append(t1)
        
        t0 = time()
        y_predict = estimator.predict(X_validate, Y_validate)
        t1 = time()-t0
        
        y_predictions.append(y_predict)        
        estimation_times.append(t1)
        time_labels.append(est_descr)
                
    graph_train_test_times(train_times, estimation_times, labels)
    graph_confusion(Y_true, y_predictions, labels, path)   
        
        
        
        
def graph_train_test_times(train_times, test_times, time_labels):
    N = len(time_labels)
    
    ind = np.arange(N)
    width = 0.35
    
    p1 = plt.bar(ind, train_times, width, color='r')
    p2 = plt.bar(ind, test_times, width, color='y', bottom = train_times)
    
    plt.ylabel('time (ms)')
    plt.title('Overall train/test times')
    plt.xticks(ind+width/2., labels, rotations=45)
    
    plt.legend( (p1[0],p2[0]), ('train', 'test') )
    
    return plt
    

    
    
def graph_confusion(Y_true, Y_predictions, labels, path):
    
    for num, y_predict in enumerate(Y_predictions):
        cm = metrics.confusion_matrix(Y_true, y_predict)
        plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
        plt.title('Confusion matrix of ' + labels[num])
        plt.colorbar()
        
        tick_marks = np.arange(len(np.unique(Y_true)))
        plt.xticks(tick_marks, np.unique(Y_true), rotation=45)
        plt.yticks(tick_marks, np.unique(Y_true))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        savefig(plt, path+'confusion/', labels[num])

    
def graph_roc_miss(X_validation, Y_validation, estimators, path, method):
    
    figures = {}
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
    
#%%
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    data = load_iris()
    X = data.data
    Y = data.target
    
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    
    estimators = [MultinomialNB(), SVC()]
    best = {}   
    for cnt, estimator in enumerate(estimators):
        estimator.fit(X, Y)
        best.update({'estim' + str(cnt):estimator})
        
    