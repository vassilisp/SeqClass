import LoadingTestData
import DBconnection
from queries import Queries

import numpy as np

def generateStatistics(processID, divider):

    X,Y = LoadingTestData.loadData(processID, divider)
    #transitions per sequence //dependent on diviser

    
    
    
    
    #transitions per day
    
    
    #transitions per user


def transPerDay(processID):
    con = DBconnection.getConnection()
    cur = con.cursor()
    q =  Queries.getTransPerDay(procesID)
    
    cur.execute(q)
    A = cur.fetchall()

def transPerSequence(Y):
    a = np.ones_like(Y)
    for i,seq in enumerate(Y):
        a[i] = seq.count(' ')
    totalTrans = a.sum()
    meanTrans = a.mean()
    stdTrans = a.std()
    
    print('total transitions:', totalTrans)
    print('mean:', meanTrans)
    print('std:', stdTrans)
    
    return a