import time

import DBconnection
import Globals

import Statistics
import LoadingTestData
from statisticsGen import StatisticsGen

import rebatcher

from basicQueries import BasicQueries


exeTime = time.strftime('%d%m_%H%M')
con = DBconnection.getConnection()

A = DBconnection.connectAndFetchArray(BasicQueries.getAllProcessIDs())
A = A[:,0]
A = ['pro307653']

q_dividers = ['clientId'] #'clientId,subSession' -- if added - add also a FOR LOOP
dividers = {0:'full', 100:'batch100' , 200:'batch200'} 
for proID in A:
    
    if __name__ == "__main__":
    
        Globals.getProcessIDPath(proID, exeTime)

        #processID general statistics
        statGen = StatisticsGen(proID,'clientId', exeTime)        
        statGen.transPerSequence();
        statGen.totalTransPerDay();      
        statGen.sequencesPerUser();
        statGen.totalTransPerUser();
        
        
        from sklearn.cross_validation import train_test_split
        X, Y = LoadingTestData.loadTestData(proID, q_dividers[0])
        
        #Held out sample for later validation
        X_develop, X_validate , Y_develop, Y_validate = train_test_split(X,Y, 0.3)
        
        for div in dividers:
            X_train, Y_train = rebatcher.single_rebatcher(X_develop, Y_develop)
            
            
            
            
            
            
            
        

    
    
    
    