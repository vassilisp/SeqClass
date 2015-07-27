import numpy as np
import pymysql as sql

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab


from astroML.plotting import hist
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=10, usetex=False)


from queries import Queries
import DBconnection
import Globals

def run(processID):
    #---------------------------
    #connet to the database and fetch results
    connection = DBconnection.getConnection()

    c = connection.cursor()

    c.execute("Set group_concat_max_len=100000;")

    "select count(refererId) as cnt from Preprocess group by userId order by cnt desc"

    processID = str(processID)
    q = Queries.getTransPerUserQ(processID)
    print(q)
    num_rows = c.execute(q)


    results = c.fetchall()
    connection.close()

    B = np.asarray(results)
    #print(A)
    A = B[:,1]
    A = A.astype(int)
    
    #---------
    #remove outliers 10%
    limit = len(A)*0.1;
    #A = A[limit:]
    #---------
    
    fig1 = plt.figure()
    num_bins = 20

    n, bins, patches = plt.hist(A, num_bins)
    plt.xlabel("xlabel")
    plt.ylabel("ylabel")

    plt.title("Histogram of transitions per user " + processID)

    #-----
    fig = plt.figure()
    #plt.xlabel("xlabel")
    #plt.ylabel("ylabel")
    #plt.title("Histogram of transitions per user")
    
    plt.suptitle("Histogram of transitions per user " + processID)
    fig.subplots_adjust(hspace=0.2, left=0.07, right=0.95, wspace=0.05, bottom=0.15)

    fig_x_dim = 1
    fig_y_dim = 1
    ax = [fig.add_subplot(fig_y_dim, fig_x_dim, i+1) for i in range(fig_x_dim*fig_y_dim)]

    #choose graph to place results
    axx = ax[0]
    bins = ['blocks', 'knuth', 'scott', 'freedman']
    bins = 'freedman'

    counts,bins, patches = hist(A, bins=bins, ax=axx)
    #axx.set_title('')

    #-------------------

    mean = A.mean()
    sigma = A.std()
    
    from reporter import Reporter
    reporter = Reporter()
    reporter.report('Transitions per User')
    reporter.report('mean: ' + str(mean))
    reporter.report("STD: " + str(sigma))
    
    x = np.linspace(bins.min(),bins.max(),100)
    y = mlab.normpdf(x,mean,sigma)
    axx.plot(x, y*counts.max(), 'r--')
    #----------

    
    fig3 = plt.figure()
    x = range(len(A))
    plt.bar(x,A, align='center')
    plt.xticks(x)
    
    path = Globals.getProcessIDPath(processID)
    
    filename = processID + '_freedman_histo_TransPerUser.svg'
    fig.savefig(path + filename, format='svg', dpi=1200)
    
    filename1 = processID + '_20binHisto_TransPerUser'
    fig1.savefig(path + filename1, format = 'svg', dpi=1200)
    
    return reporter