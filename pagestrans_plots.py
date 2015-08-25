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
q1 = """
        select concat(refererID,':', targetID), count(refererID) as cnt from Preprocess where processID= 'pro208959' group by refererID,targetID order by cnt desc
     """

q2 = """SELECT refererID AS pages, count(refererID) AS cnt 
        FROM 
        (SELECT refererID FROM Preprocess WHERE processID= 'pro208959' 
        UNION ALL
        SELECT targetID FROM Preprocess WHERE processID= 'pro208959') AS allpages
        GROUP BY refererID ORDER BY cnt DESC
        """


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

    if token == '1' or token =='2':
        result = cur.fetchall()
        A = np.asarray(result)
    
        trans = A[:,0]
        freq = A[:,1]
        freq = freq.astype('int')
        x = np.arange(len(A)) + 1
        
        freq = freq[0:60]
        plt.figure(figsize=(9,6))
        plt.bar(x,freq, align='center')
        #plt.xticks(x,trans, rotation=45, ha='right')
        plt.title('Transition Frequencies')
        plt.xlabel('transitions')
        plt.ylabel('frequency')
        
    con.close()

    
    

    con = DBconnection.getConnection()
    cur = con.cursor()
    
    pages.append(cur.execute(exe2))
    
    if token == '1' or token =='2':
        result = cur.fetchall()
        A = np.asarray(result)
    
        pag = A[:,0]
        freq = A[:,1]
        freq = freq.astype('int')
    
        x = np.arange(len(A)) + 1
        freq = freq[0:60]
        plt.figure(figsize=(9,6))
        plt.bar(x,freq, align='center')
        plt.title('Page Frequencies')
        plt.xlabel('pages')
        plt.ylabel('frequency')
        #plt.xticks(x,pag, rotation=45, ha='right')
    

    
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
ax.set_xlabel('URL partitions')
ax.set_ylabel('Unique Count')

ax.legend( (p2[0],p1[0]), ('Pages', 'Transitions'), loc='best' )

autolabel(p1)
autolabel(p2)

##TODO savefigures

from stratified_metrics import savefig
fig.savefig('pro28959_pagetrans_graph' , dpi=600, format='svg')

#%%
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')