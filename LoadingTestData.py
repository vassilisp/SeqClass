
import DBconnection
import scipy as sp
import numpy as np



def loadTestData(processID, divider, tokens):
    con = DBconnection.getConnection()
    cur = con.cursor()

    token_string = 'refererIDXxX, targetIDXxX'
    if tokens!=0:
        token_string = token_string.replace('XxX', str(tokens))
    else:
        token_string = token_string.replace('XxX', '')
    
    q = "select userId, group_concat(&&TOKENS&& order by timestamp asc separator ' ') FROM Preprocess where processID=':processID' group by :divider order by timestamp"
    
    q = q.replace('&&TOKENS&&', token_string)    
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