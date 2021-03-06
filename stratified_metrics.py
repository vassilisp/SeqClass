# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:50:17 2015

@author: vpan
"""

from sklearn.cross_validation import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

from sklearn.datasets import make_classification
from time import time
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from Globals import clfColors

import matplotlib.pyplot as plt
import sys
    
def EVALUATE_TEST(X,Y, kept_estimators, path, method):
    
    
    t0 = time()
    cv = StratifiedKFold(Y, n_folds=4,shuffle=True)
    t1 = time()-t0
    print('computed FOLDS in: ', t1)
    
    
    
    
    #%%
    
    
    from sklearn.base import clone
    
    
    
    from scipy.sparse import csr_matrix, csc_matrix
    from numpy import ndarray
        

    if not (isinstance(X, csc_matrix) or isinstance(X, csr_matrix)):
        if not isinstance(X, ndarray):
            X = np.asarray(X)            
            #X = csr_matrix(X)

    
    


    

    timebars, timebax = plt.subplots()
    timebax.hold(True)

    rocs, rocsax = plt.subplots()
    rocsax.hold(True)
    
    false, falseax = plt.subplots()
    falseax.hold(True)
    
    falseL, falseLax = plt.subplots()
    falseLax.hold(True)
    
    scor, scorax = plt.subplots()
    scorax.hold(True)
    
    labels = []
    
    p2 = []
    
    for j,(estim_descr,kestimator) in enumerate(kept_estimators.items()):
        try:
            t_train = np.zeros(len(cv))
            t_predict = np.zeros(len(cv))
            y_prediction = []
            y_real = []
            y_pb_all = []
            
            y_scores = []
            
            
            mean_tpr = 0.0
            mean_fpr = np.linspace(0,1,100)
    
            print('Starting cross validation')
            x=X
            for i, (train,test) in enumerate(cv):
                estimator = clone(kestimator)
                print('cloned estimator ready')
                print('iteration:',i)
                t0 = time()
                estimator.fit(x[train], Y[train])
                t_train[i] = time() - t0        
                            
                print(estim_descr, ' train in:',t_train[i])
                t0 = time()
                y_predict = estimator.predict(x[test])
                t_predict[i] = time()-t0
                
                y_prediction.append(y_predict)
                y_real.append(Y[test])
                
                y_pb = estimator.predict_proba(x[test])
                y_pb_all.append(y_pb)
                
                y_scores.append(estimator.score(x[test], Y[test]))
                
                #Mean ROC
                lb = LabelBinarizer().fit(Y)
                y_true = lb.transform(Y[test])
                fpr, tpr, _ = metrics.roc_curve(y_true.ravel(), y_pb.ravel())
                mean_tpr += interp(mean_fpr,fpr,tpr)
                mean_tpr[0]=0
                #+++
                
                print('predict in:', t_predict[i])
            
            labels.append(estim_descr)
    
            mean_tpr /= len(cv)
            mean_tpr[-1] = 1
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            
            #colorargs = clfColors(estimator)
            
            rocsax.plot(mean_fpr, mean_tpr, lw=1.2, label= estim_descr+' auc: %0.3f' % mean_auc)
    
            falseax.plot(mean_fpr, 1-mean_tpr, lw=1.2, label= estim_descr+' auc: %0.3f'% mean_auc )
            falseLax.plot(mean_fpr, 1-mean_tpr, lw=1.2, label= estim_descr+' auc: %0.3f' % mean_auc )
            
            train_times = t_train.mean(axis=0)
            std_train = t_train.std(axis=0)
            test_times = t_predict.mean(axis=0)
            std_test = t_predict.std(axis=0)
        
            #%%
            p1 = timebax.bar(j+0.3, train_times, yerr=std_train, ecolor='b', color='r', align='center')
            p2 = timebax.bar(j+0.3 , test_times, bottom=train_times, yerr=std_test,ecolor='g', color='y', align='center')
        
            #%%
            y_score = np.asarray(y_scores)
            y_score_mean = y_score.mean()
            y_score_std = y_score.std()
            p3 = scorax.bar(j+0.3, y_score_mean, yerr=y_score_std, ecolor='r', color='w', align='center')
            
            #%%
            y_real = np.concatenate(y_real)
            y_prediction = np.concatenate(y_prediction)
            y_pb_all = np.concatenate(y_pb_all)
        
        
            #%%
            lb = LabelBinarizer()
            y_real_bin = lb.fit_transform(y_real)
            y_prediction_bin = lb.transform(y_prediction)
        
    
        
        #%%
            fpr, tpr, thr = metrics.roc_curve(y_real_bin.ravel(), y_pb_all.ravel())
            rocsax.plot(fpr,tpr, label='pb'+estim_descr)
    
            #%%
            
            
            #%%NAIVE
            from sklearn.cross_validation import train_test_split
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2)
            estimator.fit(Xtrain,Ytrain)
            print(estimator.score(Xtest,Ytest))
            y_pb = estimator.predict_proba(Xtest)
            lb = LabelBinarizer()
            lb.fit(Y)
            y_bin = lb.transform(Ytest)
            
            fpr,tpr, _ = metrics.roc_curve(y_bin.ravel(), y_pb.ravel())
            rocsax.plot(fpr,tpr,label='wrong'+estim_descr)
            rocsax.plot([0,1],[0,1], 'b--', lw=0.8)
            
            
            graph_confusion(y_real, y_prediction, estim_descr, '')
        except:
            print(sys.exc_info())
    #%%              
    
    rocsax.legend(loc='best')
    rocsax.set_xlabel('FALSE POSITVE (%)')
    rocsax.set_ylabel('TRUE POSITIVE (%)')
    rocsax.grid(b=True ,which='major')

    #rocsax.set_xscale('log')
    try:
        timebax.legend( (p2[0],p1[0]), ('test', 'train'), loc='best' )
        timebax.set_ylabel('time (s)')
        timebax.set_xticks(np.arange(0, len(labels),1)+0.3)        
        timebax.set_xticklabels(labels, rotation=45)
    except:
        pass
    falseax.plot([0,1], [1,0], 'b--' , lw =0.6)
    falseax.set_ylabel('MISSING ALARMS (%)')
    falseax.set_xlabel('FALSE ALARMS (%)')
    falseax.legend(loc='best')
    falseax.grid(b=True ,which='major')    
    falseLax.set_ylabel('MISSING ALARMS (%)')
    falseLax.set_xlabel('FALSE ALARMS (%)')
    falseLax.legend(loc='best')
    falseLax.set_xlim([0.001,1])
    falseLax.set_xscale('log')
    falseLax.grid(b=True ,which='minor')
    falseLax.grid(b=True ,which='major')
    
    scorax.set_ylabel('ACCURACY (jaccard similarity)')
    scorax.set_xticks(np.arange(0, len(labels),1)+0.3)        
    scorax.set_xticklabels(labels, rotation=45)

    #fdescr = proID +'_' + div +'_'+ func_descr + '_'
    #savefig(fig1, path,  fdescr + 'roc_curve.svg')
    #savefig(fig2, path,  fdescr + 'missed_alarms_log.svg')
    #savefig(fig3, path,  fdescr + 'missed_alarms.svg')

#%%

def graph_bothConfusions(Y_true, Y_predict, label, path):
    pass
    
def graph_confusion(Y_true, Y_predict, label, path):

    fig,axx = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    
    ax = axx[0]
    cm = metrics.confusion_matrix(Y_true, Y_predict)
    #cm = cm/ cm.max()
    ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    ax.set_title('Confusion Matrix of ' + label)
    
    tick_marks = np.arange(len(np.unique(Y_true)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')
    fig.gca().invert_yaxis()
    
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(Y_true)
    y_pred_bin = lb.transform(Y_predict)
    
    
    ax = axx[1]
    cm = metrics.confusion_matrix(y_bin.ravel(), y_pred_bin.ravel())
    #cm = cm/ cm.max()
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    ax.set_title('Binary Matrix of ' + label)
    
    tick_marks = np.arange(len(np.unique(y_bin)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')
    
    fig.gca().invert_yaxis()
    fig.tight_layout()
    plt.colorbar()    

    
    #savefig(plt, path+'confusion/', labels)
#%%
def savefig(plt, path, filename):
    #plt.savefig(path + filename, dpi=1200, format='svg' )
    pass

    
#%%
def EVALUATE_TOPSCORES(kepttopscores, path, method):
    plt.figure()
    
    labels = []
    for i, (desc, top_scores) in enumerate(kepttopscores.items()):
        mean_scores = []
        labels.append(desc)
        for j, score in enumerate(top_scores):
            if j!=0:            
                plt.bar(i+0.3, score.mean_validation_score, bottom=mean_scores[j-1] ,yerr=np.std(score.cv_validation_scores),  ecolor='r', color='w', align='center')
            else:
                plt.bar(i+0.3, score.mean_validation_score,yerr=np.std(score.cv_validation_scores), ecolor='r', color='w', align='center')
            mean_scores.append(score.mean_validation_score)
            
    
    plt.ylabel('Scores')
    plt.xticks(np.arange(0, len(labels),1)+0.3, labels, rotation=45)
    
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    
    X,Y = make_classification(n_samples=1000, n_classes=30 ,n_informative=10, n_features=1000)
    #X = load_iris().data
    #Y = load_iris().target
    X= np.abs(X)    
    
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    mnb = make_pipeline(TfidfVectorizer(ngram_range=(3,4)), MultinomialNB())
    
    svc = OneVsRestClassifier(SVC(kernel='linear', probability=True), n_jobs=-1)
    svc01 = OneVsRestClassifier(SVC(C=0.1, kernel='linear', probability=True), n_jobs=-1)
    rsvc = OneVsRestClassifier(SVC(kernel='rbf', probability=True), n_jobs=-1)
    
    psvc01 = make_pipeline(TfidfVectorizer(ngram_range=(3,4)), svc01)    
    psvc1 = make_pipeline(TfidfVectorizer(ngram_range=(3,4)), svc)    
    rsvc = make_pipeline(TfidfVectorizer(ngram_range=(3,4)),rsvc)   

    
    
    kept_estimators = {'naiveBayes':mnb, 
                       'pSVC': psvc1,
                       'rsvc': rsvc,
                       'psvc0.1':psvc01}
    #%%
                       
    import LoadingTestData

    #X,Y = LoadingTestData.loadTestData('pro307653', 'clientId', 1)                       
    X,Y = LoadingTestData.loadTestData('pro48556', 'clientId', 0) 
    #%%
    EVALUATE_TEST(X,Y,kept_estimators, '/home/vpan/TESTSTESTTST/', 'testimator')
        