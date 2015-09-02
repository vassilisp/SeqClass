# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:17:03 2015

@author: vpan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:59:22 2015

@author: vpan
"""
import numpy as np
import DBconnection
import time as ttime

def loadTransTimesData(processID):
    
    
    
    con = DBconnection.getConnection()
    cur = con.cursor()


    dbusers = "select distinct userId FROM Preprocess where processID=':processID'"
    
    dbsessions = "Select distinct clientId FROM Preprocess WHERE processID=':processID' and userId=':userId'"
    
    q1 = "select userId, clientId, timestamp FROM Preprocess where processID=':processID'  order by userId, clientId, timestamp asc"
    q2 = "select clientId, timestamp FROM Preprocess where processID=':processID'  and userId=':userId' order by clientId, timestamp asc"
    


    dbusers = dbusers.replace(":processID", processID, 1)
    dbsessions = dbsessions.replace(":processID", processID, 1)
    q2 = q2.replace(":processID", processID, 1)
    
    print(cur.execute(dbusers))
    users = cur.fetchall()
    
    u_times = np.zeros((len(users)))
    s_times = []
    s_u_labels = []
    for u,user in enumerate(users):
        user = user[0]
        
        q22 = q2.replace(':userId', user)       
        print('user', u, cur.execute(q22))
        trans = cur.fetchall()
        s_times_tmp = []    
        
        prev_session = 0
        acc_time = 0
        prev_time = 0
        counter = 0
        for t, entry in enumerate(trans):
            session = entry[0]
            time = entry[1]

            
            if prev_session == session or prev_session == 0:
                #continue
                prev_session = session
                
                if prev_time !=0:        
                    dif = time - prev_time
                    print(u,'s',session, '---dif:', dif)
                    if dif>20*60*1000: #m*s*ms
                        print('-'*50, 'outlier')
                    else:
                        acc_time += dif
                        counter += 1
                    
                    prev_time = time
                else:
                    prev_time = time
                
            else:
                #finalize session metrics

                avg_s_time = acc_time/counter
                print('session average:', avg_s_time)
                s_times.append(avg_s_time)
                s_times_tmp.append(avg_s_time)
                s_u_labels.append(user)
                
                acc_time = 0
                prev_time = time
                prev_session = session
                counter = 0 
            
                
        sum_s = np.sum(s_times_tmp)
        avg = sum_s/len(s_times_tmp)
        print('users average', avg)
        u_times[u] = avg
    
    return u_times, s_times, s_u_labels    
    
    con.close()

#%%
if __name__ == '__main__':
    
    proID = 'pro288817'
    u_times, s_times, s_u_labels = loadTransTimesData(proID)
    
    print(u_times)
    
    #%%
    all_avg = np.average(u_times)
    print(all_avg)
    
    import matplotlib.pyplot as plt
    
    j = np.arange(len(u_times))
    
    u_times_mins = u_times*200/(60*1000)
    
    plt.figure(figsize=(9,6))
    
    plt.bar(j, u_times_mins)
    
    plt.xlabel('USERS')
    plt.ylabel('Average time -200 transitions (in minutes)')
    
    plt.savefig(proID +'_200transtime.svg', dpi=600, format='svg')