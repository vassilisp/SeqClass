# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 18:34:42 2015

@author: vpan
"""

from os import listdir
from os.path import isfile, join, dirname, abspath

import pickle
import LoadingTestData

from stratified_metrics import EVALUATE_TEST

import rebatcher
import Globals

import time
def singleEstImport(estimator, mypickle):
    name = ''        
    start = mypickle.find('_noDR_')
    if start != -1:
        start = start + len('_noDR_')
    elif mypickle.find('_TruncatedSVD_') != 0:
        start = mypickle.find('_TruncatedSVD_')
        if start!=-1:
            start = start + len('_TruncatedSVD_')
            name = 'SVD'
        else:
            print('ton ipiaME')
            exit
    elif  mypickle.find('_SOMETHING_') != 0:
        start = mypickle.find('_SOMETHING_')
        name = 'SOMETHING'        
        
    
    stop = mypickle.find('bestestimator')
    if stop ==-1:
        stop = mypickle.find('customestimator')        
        if stop == -1:
            print('ton katapiaME')
            exit
    
    tmp = mypickle[start:stop-1]
    name = tmp + ' ' + name
    return name        
    

def loadclassifiers():
    mypath = dirname(dirname(abspath(__file__)))
    
    loadpath = mypath + '/CLASSIFIERS'
    onlyfiles = [ f for f in listdir(loadpath) if isfile(join(loadpath,f)) ]
    
    print(onlyfiles)
    
    pickles = [ f for f in onlyfiles if f.endswith('.pickle')]
    
    loaded = {}
    for mypickle in pickles:
        
        picklepath = join(loadpath, mypickle)
        with open(picklepath, 'rb') as infile:
            est = pickle.load(infile)
        
        if est.__class__.__name__ == 'Pipeline':
            #figure name for the dictionary entry
            print('only one estimator')
            
            name = singleEstImport(est, mypickle)
            est = {name: est}
        else:
            #import a complete dictionary and figure out what to do with it
            print('estimator dictionary')
            print('create the inport functions')
            
        
        
        loaded.update(est)
        
    print(loaded)
    return loaded

if __name__ == '__main__':
    
    classifierslist = loadclassifiers()
    proIDs = ['pro288817']#'pro288817','pro288955'
    for proID in proIDs:
    #load X,Y and run test
        X, Y = LoadingTestData.loadTestData(proID, 'clientId',2)
    
        X, Y, reporter = rebatcher.single_rebatcher(X,Y, 200)
    
        #%%
        ##NAME YUOR METHOD
        method = 'T2-200-w' +proID
        #%%
        
        exeTime = time.strftime('%d%m_%H%M')
        path = Globals.getResultsPATH()
    
        path = Globals.getProcessIDPath(method, exeTime)
    
        EVALUATE_TEST(X,Y, classifierslist, path, method)
        
    
    