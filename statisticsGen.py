from queries import Queries
import Globals
from LoadingTestData import loadTestData
import numpy as np
import DBconnection
import totalTransitionsPerUserHistogram
from reporter import Reporter

class StatisticsGen:
    processID = ''
    divider = ''
    exeTime = ''
    X = []
    Y = []
    reporter = Reporter()
    
    def __init__(this, processID, divider, exeTime):
        this.processID = processID
        this.divider = divider
        this.exeTime = exeTime
        this.reporter.new_report("Statistics of " + processID + " at " + exeTime)
        
        this.X, this.Y = loadTestData(processID, divider)
        
    def savefig(this, fig, filename):
        path = Globals.getProcessIDPath(this.processID, this.exeTime)
        filename = this.processID + '_' + filename + '.svg'
        
        fig.savefig(path + filename, format = 'svg', dpi=1200)
        
    def saveReport(this, report, filename):
        path = Globals.getProcessIDPath(this.processID, this.exeTime)
        filename = this.processID + '_report_' + filename
        
        #@TODO save file


    def getPath(this):
        return Globals.getProcessIDPath(processID)
        

    def totalTransPerUser(this):
        report = totalTransitionsPerUserHistogram.run(this.processID, this.exeTime)
        this.reporter.concat_report(report)



    def totalTransPerDay(this):
        q = Queries.getTransPerDayQ(this.processID)
        print(q)
        
        result = DBconnection.connectAndFetchArray(q)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        A = result[:,0].astype(str)
        B = result[:,1].astype(int)
        
        x = np.arange(1,len(A)+1)
        rects = ax.bar(x,B, align='center')
        ax.set_ylabel('Transitions')
        ax.set_xlabel('days')
        ax.set_title("Total Transitions per Day")
        
        #add 0.5 to x
        ax.set_xticks(x)
        ax.set_xticklabels(A, rotation=45)
        
        this.reporter.subreport('Total transitions per day')
        for day in zip(A,B):        
            this.report.report(str(day))
            
        this.savefig(fig, 'totalTransitionsperDay')


    def transPerSequence(this,X=None, Y=None):
        if (X == None or Y == None):
            X = this.X
            Y = this.Y
        
        a = np.ones_like(Y)
        for i,seq in enumerate(X):
            a[i] = seq.count(' ')
            
        a = a.astype(int)
        totalTrans = a.sum()
        meanTrans = a.mean()
        stdTrans = a.std()
        
        this.reporter.subreport('Tranitions per Sequence')
        this.reporter.report('total transitions per sequence:' + str(totalTrans))
        this.reporter.report('mean:' + str(meanTrans))
        this.reporter.report('std:' + str(stdTrans))
        
        import matplotlib.pyplot as plt
        
        ticks = np.zeros(len(np.unique(Y)))
        for i, c in enumerate(np.unique(Y)):
            c_len = len(this.Y[Y==c])
            if i==0:
                ticks[i] = c_len
            else:
                ticks[i] = ticks[i-1] + c_len
            
        fig = plt.figure()
        x = np.arange(1,len(a)+1)
        plt.bar(x,a, align = 'center', color='b')
        
        plt.vlines(ticks[:-1],0,plt.ylim()[1],color='r', linestyles='solid')
        plt.xlabel("clientIds")
        plt.ylabel("transitions")
        plt.xlim = (0,len(a))
        plt.title('Transitions per Sequence')
        this.savefig(fig, 'Transitions_per_Sequence')
        return a

    def sequencesPerUser(this):
        #to TEST
        cnt = np.zeros(len(np.unique(this.Y)))
        for i, c in enumerate(np.unique(this.Y)):
            cnt[i] = (len(this.Y[this.Y==c]))
        
        import matplotlib.pyplot as plt
        fig = plt.figure()

        x = np.arange(1,len(cnt)+1)
        plt.bar(x, cnt, align='center')
        plt.ylabel('Number of Sequences')
        plt.xlabel('User')
        plt.title('Sequences per User')
        plt.xticks(x)
        plt.xlim = (0,x.max())
        this.savefig(fig, 'sequencesPerUser_' + this.divider)
        
        this.reporter.subreport('Sequences per user')
        for tmp in cnt:        
            this.reporter.report(str(tmp))
        
    def transPerUserPerClientId(this):
        #to TEST
        q = Queries.getTransPerUserPerClientId(this.processID)
        
        result = DBconnection.connectAndFetchArray(q)
        
        users = result[:,0]
        cids = result[:,1]
        counts = result[:,2]
        
        counts = counts.astype(int)
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        plt.title('Transitions per User per ClientId')
        plt.xlabel('ClientIds')
        plt.ylabel('Number of Transitions')
        x = np.arange(1,len(cids)+1)
        ax.bar(x, counts)
        
        
        plt.show()
        
        
        

    def getReport(this):
        return this.report
        
