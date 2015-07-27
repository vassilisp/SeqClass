
import totalTransitionsPerUserHistogram
import DBconnection
import Globals

import Statistics
import LoadingTestData
from statisticsGen import StatisticsGen

from basicQueries import BasicQueries



con = DBconnection.getConnection()

A = DBconnection.connectAndFetchArray(BasicQueries.getAllProcessIDs())
A = A[:,0]
A = ['pro32240']


dividers = ['clientId', 'clientId,subSession', 'custom']
for proID in A:
    
    Globals.mkdir_p(proID)
    
    reporter1 = totalTransitionsPerUserHistogram.run(proID)
    
    statGen = StatisticsGen(proID, 'clientId')
    
    statGen.transPerSequence();
    statGen.totalTransPerDay();
    