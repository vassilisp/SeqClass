# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:32:42 2015

@author: vpan
"""
import DBconnection
import matplotlib.pyplot as plt
import numpy as np

dic = {'1':'refererID1', '2':'refererID2', '3':'refererID3', '4':'refererID4', '0':'refererID'}

tokens = ['1','2','3','4', '0']

proID = 'pro288817'

q1 = """
        select concat(refererID,':', targetID), count(refererID) as cnt from Preprocess where processID=':processID' group by refererID,targetID order by cnt desc
     """

#q2 verified twice to be a correct query - dont worry again
q2 = """SELECT refererID AS pages, count(refererID) AS cnt 
        FROM 
        (SELECT refererID FROM Preprocess WHERE processID=':processID'
        UNION ALL
        SELECT targetID FROM Preprocess WHERE processID=':processID') AS allpages
        GROUP BY refererID ORDER BY cnt DESC
        """

q1 = q1.replace(':processID', proID)
q2 = q2.replace(':processID', proID)

pages = []
transitions = []
labels = []
for token in tokens:
    
    if token!='0':        
        exe1 = q1.replace('refererID', 'refererID' + str(token))
        exe2 = q2.replace('refererID', 'refererID' + str(token))
        
        exe1 = exe1.replace('targetID', 'targetID' + str(token))
        exe2 = exe2.replace('targetID', 'targetID' + str(token))
    else:
        exe1 = q1
        exe2 = q2
    
    con = DBconnection.getConnection()
    cur = con.cursor()
    transitions.append(cur.execute(exe1))

    cut0 = 2
    cut1 = 100
    if token == '1' or token =='2' or 1 == 1:
        result = cur.fetchall()
        A = np.asarray(result)
    
        trans = A[:,0]
        freq = A[:,1]
        freq = freq.astype('int')
        x = np.arange(len(A)) + 1
        
        freq = freq[cut0:cut1]
        x = x[cut0:cut1]
        fig, ax = plt.subplots(figsize=(12,6))
        plt.bar(x,freq, align='center')
        #plt.xticks(x,trans, rotation=45, ha='right')
        plt.title('Transition Frequencies (tokens=' + str(token) + ')')
        plt.xlabel('transitions')
        plt.ylabel('frequency')
        plt.autoscale(False)
        ax.set_xlim(cut0,cut1+1)
        form = 'svg'
        fig.savefig(str(proID) + '_trans_freq_t' + str(token) + '.' + form  , dpi=600, format=form)        
        
    con.close()

    
    

    con = DBconnection.getConnection()
    cur = con.cursor()
    
    pages.append(cur.execute(exe2))
    
    if token == '1' or token =='2' or 1 == 1:
        result = cur.fetchall()
        A = np.asarray(result)
    
        pag = A[:,0]
        freq = A[:,1]
        freq = freq.astype('int')
    
        x = np.arange(len(A)) + 1
        freq = freq[cut0:cut1]
        x = x[cut0:cut1]
        fig, ax = plt.subplots(figsize=(12,6))
        plt.bar(x,freq, align='center')
        plt.title('Page Frequencies  (tokens=' + str(token) + ')')
        plt.xlabel('pages')
        plt.ylabel('frequency')
        plt.autoscale(False)
        ax.set_xlim(cut0,cut1+1)

        #plt.xticks(x,pag, rotation=45, ha='right')
        form = 'svg'
        fig.savefig(str(proID) + '_page_freq_t' + str(token) + '.' + form  , dpi=600, format=form)

    
    con.close()
    
    labels.append(str(token))
#%%
    
x = np.arange(len(dic)) + 1

fig, ax = plt.subplots(figsize=(9,6))
width=0.35

p2 = ax.bar(x, pages, width, align='center',color='y')
#ax.set_xticks(x, labels)

#plt.figure()
p1 = plt.bar(x + width, transitions, width, align='center', color='b')
ax.set_xticks(x + width/2)
ax.set_xticklabels(labels)
ax.set_xlabel('URL tokens')
ax.set_ylabel('Unique Count')

ax.legend( (p2[0],p1[0]), ('Pages', 'Transitions'), loc='upper left' )

autolabel(p1)
autolabel(p2)

##TODO savefigures

form = 'svg'
fig.savefig(str(proID) + '_pagetrans_graph.' + form  , dpi=600, format=form)

#%%
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')