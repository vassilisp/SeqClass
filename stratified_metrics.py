# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:50:17 2015

@author: vpan
"""

import traceback

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

from Globals import clfColors, mkdir_LR

import matplotlib.pyplot as plt
import sys

import json

markers = ['-','-+','--', '-.',':', ':+','-x','--x','--o','--+','-->','-o','->','--D','-*','-D']
import itertools

#%%
def EVALUATE_TEST(X,Y, kept_estimators, path, method):
    styles = itertools.cycle(markers)
    
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

    
    


    

    timebars, timebax = plt.subplots(figsize=(8,6))
    timebax.hold(True)

    rocs, rocsax = plt.subplots(figsize=(8,6))
    rocsax.hold(True)
    
    false, falseax = plt.subplots(figsize=(8,6))
    falseax.hold(True)
    
    falseL, falseLax = plt.subplots(figsize=(8,6))
    falseLax.set_xscale('log')
    falseLax.hold(True)
    
    scor, scorax = plt.subplots()
    scorax.hold(True)
    
    fmeasure, fmeasurax = plt.subplots(figsize=(8,6))
    fmeasurax.hold(True)
    
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
            left_fpr = 100000.0
            mean_fpr = np.linspace(0,1,left_fpr)
    
            print('Starting cross validation of ', estim_descr)
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
                
                if hasattr(estimator, 'decision_function'):
                    y_pb = estimator.decision_function(x[test])
                else:
                    y_pb = estimator.predict_proba(x[test])
                    #making scoring by roc easier
                    y_pb = y_pb * 1000
                
                y_pb_all.append(y_pb)
                
                y_scores.append(estimator.score(x[test], Y[test]))
                
                #Mean ROC
                lb = LabelBinarizer().fit(Y)
                y_true = lb.transform(Y[test])
                
                ##OUCH correcting for masquerade instead of self/noself
                y_true = 1 - y_true
                y_pb =  - y_pb
                                
                
                
                fpr, tpr, thr = metrics.roc_curve(y_true.ravel(), y_pb.ravel())

                tt = 0
                for cc, (ii, t) in enumerate(zip(fpr,tpr)):
                    if ii == 0:
                        tt = t
                    else:
                        break;
                fpr = fpr[cc-1:]
                tpr = tpr[cc-1:]
                tpr[0] = tt
                        
                mean_tpr += interp(mean_fpr,fpr,tpr)
                mean_tpr[0]=0
                #rocsax.plot(fpr, tpr, lw=0.2, label= 'my' + str(i) +'-'+ estim_descr)
                #+++
                
                print('predict in:', t_predict[i])
            
            labels.append(estim_descr)
    
            mean_tpr /= len(cv)
            mean_tpr[-1] = 1# + left_fpr_i
            mean_tpr[0] = 0
            mean_fpr[0] = 0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            
            #colorargs = clfColors(estimator)
            
            cst = next(styles)

#%%get mean thr
            
            #%%
            
            y_real = np.concatenate(y_real)
            y_prediction = np.concatenate(y_prediction)
            y_pb_all = np.concatenate(y_pb_all)
    
                
            #%%
                
                        
            lb = LabelBinarizer()
            lb.fit(Y)                        
            y_real_bin = lb.transform(y_real)
            
            #y_prediction_bin = lb.transform(y_prediction)

            masq_y_real_bin = 1 - y_real_bin
            masq_y_pb_all =  - y_pb_all            
            mfpr,mtpr, mthr = metrics.roc_curve(masq_y_real_bin.ravel(), masq_y_pb_all.ravel())

                
            mauc = metrics.auc(mfpr,mtpr)
            maucbl = auc_bylimit(mfpr,mtpr, limit=0.01)            
            
            #testing again
            mfpr[0] = 0
            mtpr[0] = 0
            mtpr[-1] = 1
            rocsax.plot(mfpr, mtpr, cst, lw=1, label= estim_descr+' auc: %0.3f / %0.10f' % (mauc, maucbl), markevery=0.1)
            
            mean_tpr = mtpr
            mean_fpr = mfpr
            
            tosave = [mfpr.tolist(), mtpr.tolist(), mthr.tolist()]
            rocDATA = tosave.copy()
            savejson(tosave, path, method + '_' + estim_descr + '___threshold_DATA')
            
#%%
            #rocsax.plot(mean_fpr, mean_tpr,cst, lw=1.2, label= estim_descr+' auc: %0.3f' % mean_auc, markevery=0.1)
            #tosave = [mean_fpr.tolist(), mean_tpr.tolist()]
            #savejson(tosave,path, method + '_' + estim_descr + '___roccurve_DATA')
    
            mean_ma = 1-mean_tpr
            mean_ma[0] = 1
            mean_ma[-1] = 0
            falseax.plot(mean_fpr, mean_ma,cst, lw=1, label= estim_descr+' auc: %0.3f'% mean_auc, markevery=0.1 )
            falseLax.plot(mean_fpr, mean_ma,cst, lw=1, label= estim_descr+' auc: %0.3f' % mean_auc, markevery=0.1 )
            ttt = 1-mean_tpr
            tosave = [mean_fpr.tolist(), ttt.tolist()]        
            savejson(tosave, path, method + '_' + estim_descr + '___missedalarms_DATA')
            
            train_times = t_train.mean(axis=0)
            std_train = t_train.std(axis=0)
            test_times = t_predict.mean(axis=0)
            std_test = t_predict.std(axis=0)
            
            tosave = [train_times.tolist(), std_train.tolist(), test_times.tolist(), std_test.tolist()]
            savejson(tosave, path, method + '_' + estim_descr + '___traintesttimes_DATA')
        
            #%%
            p1 = timebax.bar(j+0.3, train_times, yerr=std_train, ecolor='b', color='r', align='center')
            p2 = timebax.bar(j+0.3 , test_times, bottom=train_times, yerr=std_test,ecolor='g', color='y', align='center')
        
            #%%
            y_score = np.asarray(y_scores)
            y_score_mean = y_score.mean()
            y_score_std = y_score.std()
            p3 = scorax.bar(j+0.3, y_score_mean, yerr=y_score_std, ecolor='r', color='w', align='center')
            
            tosave = [y_score_mean.tolist(), y_score_std.tolist()]
            savejson(tosave, path, method + '_' + estim_descr + '___scores_DATA')
            

        
            #%%
            #DONE code to create the Fmeasure plots
            beta = np.arange(0.001,4, 0.05)
            results = []
            """
            for b in beta:
                res = metrics.fbeta_score(y_real, y_prediction, b, average='weighted')
                results.append(res)                
            
            
            fmeasurax.plot(beta, results, cst, label=estim_descr, markevery=0.1)
            
            tosave = [beta.tolist(), results]
            savejson(tosave, path, method + '_' + estim_descr + '___fmeasure_DATA')
            """
        #%%
            """
            fpr, tpr, thr = metrics.roc_curve(y_real_bin.ravel(), y_pb_all.ravel())
            rocsax.plot(fpr,tpr, label='pb'+estim_descr)
            """
            #%%
            
            """
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
            """
            
            graph_confusion(y_real, y_prediction, estim_descr, path)
            graph_confusion2(y_real, y_prediction, estim_descr, path)
            
            graph_confusion_pb(y_real_bin, y_pb_all, estim_descr, path, thresh=-1)
            #graph_confusion_pb(y_real_bin, y_pb_all, estim_descr, path, thresh=0)
            #graph_confusion_pb(y_real_bin, y_pb_all, estim_descr, path, thresh=0.0733)
            #graph_confusion_pb(y_real_bin, y_pb_all, estim_descr, path, thresh=0.1)
            
            thr_byfpr = predict_by_fpr(y_real_bin, y_pb_all, target_fpr=0.01)
            if thr_byfpr != -1:
                graph_confusion_pb(y_real_bin, y_pb_all, estim_descr + '_bytargetfpr', path, thresh=thr_byfpr)
            
        except Exception as e:
            print(sys.exc_info())
            print(e)
            print('-'*50)
            print(traceback.print_stack())
            print(traceback.print_exc())
            
            
    #%%              
    yticks = np.arange(0,1,0.1)
    
    rocsax.legend(loc='best')
    rocsax.plot([0,1], [0,1], 'b--', lw=0.6)
    #rocsax.set_xlim([0,1])
    rocsax.set_ylim([0,1])
    rocsax.set_xlabel('FALSE POSITIVE (%)')
    rocsax.set_ylabel('TRUE POSITIVE (%)')
    rocsax.grid(b=True ,which='major')
    rocsax.set_yticks(yticks)
    rocs.tight_layout()

    #rocsax.set_xscale('log')
    try:
        timebax.legend( (p2[0],p1[0]), ('test', 'train'), loc='best' )
        timebax.set_ylabel('time (s)')
        timebax.set_xticks(np.arange(0, len(labels),1)+0.3)        
        timebax.set_xticklabels(labels, rotation=15, ha='right')
        timebars.tight_layout()
    except:
        pass
    falseax.plot([0,1], [1,0], 'b--' , lw =0.6)
    falseax.set_ylabel('MISSING ALARMS (%)')
    falseax.set_xlabel('FALSE ALARMS (%)')
    falseax.legend(loc='best')
    #falseax.set_xlim([0,1])
    falseax.set_ylim([0,1])
    falseax.set_yticks(yticks)
    falseax.grid(b=True ,which='major')
    false.tight_layout()
    
    falseLax.set_ylabel('MISSING ALARMS (%)')
    falseLax.set_xlabel('FALSE ALARMS (%)')
    falseLax.legend(loc='upper right')
    falseLax.set_xlim([0.001,1])
    falseLax.set_ylim([0,1])
    falseLax.set_yticks(yticks)
    falseLax.set_xscale('log')
    falseLax.grid(b=True ,which='minor')
    falseLax.grid(b=True ,which='major')
    falseL.tight_layout()
    
    scorax.set_ylabel('ACCURACY (jaccard similarity)')
    scorax.set_xticks(np.arange(0, len(labels),1)+0.3)        
    scorax.set_xticklabels(labels, rotation=20, ha='right')
    scor.tight_layout()
    
    savefig(rocs, path,  method + '_roccurve')
    savefig(falseL, path,  method + '_missedalarms_log')
    savefig(false, path,  method + '_missedalarms')
    
    rocsax.set_xlim([0.001,1])
    rocsax.set_ylim([0,1])
    del rocsax.lines[-1]
    rocsax.legend()
    rocsax.legend(loc='lower right')
    rocsax.set_yticks(yticks)
    rocsax.set_xscale('log')
    rocsax.grid(b=True, which='minor')
    savefig(rocs, path,  method + '_roc_curvelog')
    
    savefig(scor, path, method + '_scores')
    savefig(timebars, path, method + '_traintesttimes')
    
    fmeasurax.set_ylabel("F-Beta")
    fmeasurax.set_xlabel('F-b measure')
    fmeasurax.legend(loc='best')
    fmeasurax.set_xlim([0,4.5])
    fmeasure.tight_layout()
    savefig(fmeasure, path, method + '_fmeasure')
    
    #plt.close('all')
    

#%%

def auc_bylimit(fpr,tpr, limit=1):
    keep_i = -1
    for i, cfpr in enumerate(fpr):        
        if cfpr<=limit:
            keep_i = i
        else:
            break
    nfpr = fpr[:keep_i]
    ntpr = tpr[:keep_i]
    
    auc =  metrics.auc(nfpr,ntpr)
    return auc
    

def graph_bothConfusions(Y_true, Y_predict, label, path):
    pass
    
def graph_confusion(Y_true, Y_predict, label, path):

    fig,axx = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    
    ax = axx[0]
    
    cm = metrics.confusion_matrix(Y_true.ravel(), Y_predict.ravel())
    #cm = cm/ cm.max()
    print(cm)
    
    savejson(cm.tolist(), path+'confusion/', label)    
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
    print(cm)
    savejson(cm.tolist(), path+'confusion/', label + '_bin')
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

    cmsum = cm.sum(axis=1)
    newcm = cm.T/cmsum
    newcm = newcm.T
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.annotate(str(cm[i,j]) + '\n' + '('+ str(round(newcm[i,j],3)) + ')', xy=(j, i), xytext=(j, i), ha='center', va='center')
            
    
    savefig(plt, path+'confusion/', label)

def graph_confusion_pb(Y_true, Y_pb, label, path, thresh = 0):
    
    if thresh!=-1:
        if thresh == 0:
            thresh = 0.05
        
        Y_predict = np.zeros_like(Y_true)
        Y_predict[Y_pb>=thresh] = 1

        label = 't' + str(round(thresh*100,3)) + label
        graph_bin_confusion(Y_true, Y_predict, label, path, masquerade=False)
        graph_bin_confusion(Y_true, Y_predict, label, path, masquerade=True)
        
    else:
        #Y_predict = np.zeros_like(Y_true)              
        Y_mean = Y_pb.mean(axis=1)
        Y_predict= np.zeros_like(Y_true)
        for i, (Y_pb_row, Y_mean_element) in enumerate(zip(Y_pb, Y_mean)):
            Y_predict_row = np.zeros_like(Y_pb_row)            
            Y_predict_row[Y_pb_row>Y_mean_element] = 1
            
            Y_predict[i]= Y_predict_row
            
        graph_bin_confusion(Y_true,Y_predict, label+'_meanmethod', path)        
        graph_bin_confusion(Y_true,Y_predict, label+'_meanmethod', path, masquerade=True)


def predict_by_fpr(Y_real_bin, Y_pb_all, target_fpr):
        masq_y_real_bin = 1 - Y_real_bin
        masq_y_pb_all =  - Y_pb_all
        
        #rocfpr, roctpr, rocthr = metrics.roc_curve(Y_real_bin.ravel(), Y_pb_all.ravel())
        rocfpr, roctpr, rocthr = metrics.roc_curve(masq_y_real_bin.ravel(), masq_y_pb_all.ravel())

        keep_i = -1       
        for i, fpr in enumerate(rocfpr):
            if round(fpr,3)<=target_fpr:
                keep_i = i
            else:                
                break
        if keep_i != -1:
            thr = rocthr[keep_i]
            tpr = roctpr[keep_i]
            fpr = rocfpr[keep_i]
            print(fpr, tpr, thr)
            return -thr
        else:
            return -1
                
    
def graph_bin_confusion(Y_true, Y_predict, label, path, masquerade=False):
    
    fig, ax = plt.subplots(figsize=(8,6))



    if masquerade==True:
        ccmap = plt.cm.Reds
        Y_true = 1 - Y_true
        Y_predict = 1 - Y_predict
        label += '_masq'
    else:
        ccmap = plt.cm.Blues
    cm = metrics.confusion_matrix(Y_true.ravel(), Y_predict.ravel())
    
    cmsum = cm.sum(axis=1)
    newcm = cm.T/cmsum
    newcm = newcm.T

        
    savejson(newcm.tolist(), path+'confusion2/', label + '_bin')
    plt.imshow(newcm, interpolation='nearest', cmap = ccmap, vmin=0.0, vmax=1.0)
    ax.set_title('Binary Matrix of ' + label)

    tick_marks = np.arange(len(np.unique(Y_true)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')

    fig.gca().invert_yaxis()
    fig.tight_layout()
    plt.colorbar()    

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.annotate(str(cm[i,j]) + '\n' + '('+ str(round(newcm[i,j],3)) + ')', xy=(j, i), xytext=(j, i), ha='center', va='center')

    savefig(plt, path+ 'my/', label)


def graph_confusion2(Y_true, Y_predict, label, path):

    fig,axx = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    
    ax = axx[0]
    
    cm = metrics.confusion_matrix(Y_true.ravel(), Y_predict.ravel())
    #cm = cm/ cm.max()
    
    cmsum = cm.sum(axis=1)
    newcm = cm.T/cmsum
    cm = newcm.T
    
    savejson(cm.tolist(), path+'confusion2/', label)
    ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues, vmin=0.0, vmax=1.0)
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
    
    cmsum = cm.sum(axis=1)
    newcm = cm.T/cmsum
    newcm = newcm.T
    
    savejson(newcm.tolist(), path+'confusion2/', label + '_bin')
    plt.imshow(newcm, interpolation='nearest', cmap = plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.set_title('Binary Matrix of ' + label)
    
    tick_marks = np.arange(len(np.unique(y_bin)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')
    
    fig.gca().invert_yaxis()
    fig.tight_layout()
    plt.colorbar()    

    for i in range(len(cm)):
        for j in range(len(cm)):
                ax.annotate(str(cm[i,j]) + '\n' + '('+ str(newcm[i,j])[:5] + ')', xy=(j, i), xytext=(j, i), ha='center', va='center')
    
    savefig(plt, path+'confusion2/', label)

#%%

def savejson(file, path, filename):
    if __name__ == "__main__":
        return
        pass
        
    try:
        mkdir_LR(path)    
        with open(path+filename + '.txt', 'w') as outfile:
            json.dump(file, outfile)
    except Exception as e:
            print(sys.exc_info())
            print(e)
            print('-'*50)
            print(traceback.print_stack())
            print(traceback.print_exc())
            print('ERROR SAVING')
def savefig(plt, path, filename):
    if __name__ == "__main__":
        return
        
    mkdir_LR(path)
    form = 'png'
    plt.savefig(path + filename + '.' + form, dpi=600, format=form )
    

    
#%%
def EVALUATE_TOPSCORES(kepttopscores, path, method):
    plt.figure()
    
    labels = []
    for i, (desc, top_scores) in enumerate(kepttopscores.items()):
        cum_mean_scores = 0
        labels.append(desc)
        for score in top_scores:
            plt.bar(i+0.3, score.mean_validation_score, bottom=cum_mean_scores ,yerr=np.std(score.cv_validation_scores),  ecolor='r', color='w', align='center')
            cum_mean_scores += score.mean_validation_score
            
    
    plt.ylabel('Scores')
    plt.xticks(np.arange(0, len(labels),1)+0.3, labels, rotation=20, ha='right')
    plt.tight_layout()
    
    savefig(plt, path,  method + '5topscores')
    
    plt.close()
    
    #%%
def loadandrun(path):
    import glob
    search_path = path + '*.pickle'
    #print glob.glob()
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    
    X,Y = make_classification(n_samples=1000, n_classes=30 ,n_informative=10, n_features=200)
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
                       'pSVC': psvc1
                       #'rsvc': rsvc,
                       #'psvc0.1':psvc01
                       }
                       
    
    #%%
                       
    import LoadingTestData

    #X,Y = LoadingTestData.loadTestData('pro307653', 'clientId', 1)                       
    X,Y = LoadingTestData.loadTestData('pro48556', 'clientId', 1) 
    #%%
    EVALUATE_TEST(X,Y,kept_estimators, '/home/vpan/TESTSTESTTST/', 'testimator')
        