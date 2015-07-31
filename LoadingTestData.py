
import DBconnection
import scipy as sp
import numpy as np



def loadTestData(processID, divider):
    con = DBconnection.getConnection()
    cur = con.cursor()

    q = "select userId, group_concat(refererID,targetID order by timestamp asc separator ' ') FROM Preprocess where processID=':processID' group by :divider order by userId, timestamp"
    
    q = q.replace(":processID", processID, 1).replace(":divider", divider, 1)

    print(cur.execute("Set group_concat_max_len=100000;"))
    print(cur.execute(q))

    result = cur.fetchall()
    con.close()

    A = np.asarray(result)
    #----------------------------------
    labels = A[:,0]
    sequencies = A[:,1]
    #----------------------------------
    
    return sequencies, labels

    
if __name__ == '__main__':
    
    sequencies, labels = loadTestData('pro307532', 'clientId')
    
    print(labels.shape, sequencies.shape)