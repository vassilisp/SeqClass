
import os,errno
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

from sklearn.base import BaseEstimator
from sklearn.base import clone
def mkdir_LR(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Creating PATH:', path)
    return path + '/'
    
def mkdir_RR(folder):
    directory = getResultsPATH() + folder
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('<<<<<<<><> CREATING PATH:', directory)
    return directory + '/'


def getResultsPATH():
    folder = "RESULTS"
    
    dirname = os.path.dirname
    
    
    directory = os.path.join(dirname(dirname(__file__)), folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory + '/'
    
def getProcessIDPath(processID, exeTime):
    path = processID + '_'+ exeTime + '/'
    full_path = mkdir_RR(path)
    
    return full_path
    
def clfColors(clf):
    colorarg = 'k:'
    try:
        try:
            a = clone(clf)
            while not isinstance(a,BaseEstimator):
                a = a.estimator
            if isinstance(a, OneVsRestClassifier):
                a = a.estimator
            if isinstance(a, Pipeline):
                a = a.estimator
        except:
            pass
    
    
        if isinstance(clf, MultinomialNB):
            colorarg = 'b-.'
        elif isinstance(clf, SVC):
            colorarg = 'm-'
        elif isinstance(clf, LinearSVC):
            colorarg = 'r-'
        elif isinstance(clf, SVC):
            colorarg = 'c-'
        elif isinstance(clf, SVC):
            colorarg = 'y-'
        elif isinstance(clf, SVC):
            colorarg = 'b--'
        elif isinstance(clf, SVC):
            colorarg = 'm--'
            
    except:
        pass
    return colorarg
        