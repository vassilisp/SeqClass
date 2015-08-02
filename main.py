import time

import DBconnection
import Globals

import LoadingTestData
from statisticsGen import StatisticsGen

import rebatcher
from reporter import Reporter
from basicQueries import BasicQueries


exeTime = time.strftime('%d%m_%H%M')
con = DBconnection.getConnection()

A = DBconnection.connectAndFetchArray(BasicQueries.getAllProcessIDs())
A = A[:,0]
A = ['pro307653']


#%% Setup env
generate_processID_stats = False

#%%

def main():

    q_dividers = ['clientId'] #'clientId,subSession' -- if added - add also a FOR LOOP
    dividers = {0:'full', 100:'batch100' , 200:'batch200', 300:'batch300', 400:'batch400'} 
    
    for proID in A:
            
        if __name__ == "__main__":
            
            reporter = Reporter()
            reporter.new_report("REPORT OF PROCESS: " + proID)
            
            #create and get ProcessIDPath
            proID_path = Globals.getProcessIDPath(proID, exeTime)
    
            if(generate_processID_stats == True):
                #processID general statistics
                statGen = StatisticsGen(proID,'clientId', exeTime)        
                statGen.transPerSequence();
                statGen.totalTransPerDay();      
                statGen.sequencesPerUser();
                statGen.totalTransPerUser();
                reporter.concat_report(statGen.getReport)
            
            for tokens in tokenList:
                from sklearn.cross_validation import train_test_split
                X, Y = LoadingTestData.loadTestData(proID, q_dividers[0], tokens)
                
               runTestwithDividers(X,Y, dividers, proID_path)
                
            
            
def runTestwithDividers(X,Y, dividers, proID_path):
     #Held out sample for later validation
    X_develop, X_validate , Y_develop, Y_validate = train_test_split(X,Y, test_size=0.3)
    
    for div in dividers:
        
        #create batch folder
        div_path = Globals.mkdir_LR(proID_path+dividers[div])
        
        X_develop, Y_develop, report_develop = rebatcher.single_rebatcher(X_develop, Y_develop, div)
        X_validate, Y_validate, report_validate = rebatcher.single_rebatcher(X_validate, Y_validate, div)
        report_develop.concat_report(report_validate)
        report_develop.saveReport(div_path + 'report' + div + '.txt')            
        
        reporter.concat_report(report_develop)
        
        #some stats on the new divided sets
                
        #GridSearch
        
        #score paint report save
        
        
        reporter.saveReport(proID_path + 'full_report.txt')
            
            
            
            
            
            
            
            
        

    
    
    
    