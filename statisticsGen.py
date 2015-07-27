from queries import Queries
import Globals
from LoadingTestData import loadTestData
import numpy as np
import DBconnection

from reporter import Reporter
class StatisticsGen:
    processID = ''
    divider = ''
    X = []
    Y = []
    reporter = Reporter()
    
    def __init__(this, processID, divider):
        this.processID = processID
        this.divider = divider
        
        this.X, this.Y = loadTestData(processID, divider)
        
    def savefig(this, fig, filename):
        path = Globals.getProcessIDPath(this.processID)
        filename = this.processID + '_' + filename
        
        fig.savefig(path + filename, format = 'svg', dpi=1200)
        
    def totalTransPerUserHistogram(this):
        totalTransitionsPerUserHistogram(processID)
        
    def getPath(this):
        return Globals.getProcessIDPath(processID)
        
    def totalTransPerDay(this):
        q = Queries.getTransPerDayQ(this.processID)
        print(q)
        
        result = DBconnection.connectAndFetchArray(q)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        A = result[:,0].astype(str)
        B = result[:,1].astype(int)
        
        x = np.arange(len(A))
        rects = ax.bar(x,B, align='center')
        ax.set_ylabel('Transitions')
        ax.set_title("Total Transitions")
        
        ax.set_xticklabels(A)
        
        this.savefig(fig, 'Total Transitions per Day')
        
        
        
    def transPerSequence(this):
        a = np.ones_like(this.Y)
        for i,seq in enumerate(this.Y):
            a[i] = seq.count(' ')
            
        a = a.astype(int)
        totalTrans = a.sum()
        meanTrans = a.mean()
        stdTrans = a.std()
        
        this.reporter.new_report('Total transitions per Sequence')
        this.reporter.report('total transitions:' + str(totalTrans))
        this.reporter.report('mean:' + str(meanTrans))
        this.reporter.report('std:' + str(stdTrans))
        
        return a
        
    def getReport(this):
        return this.report
        
