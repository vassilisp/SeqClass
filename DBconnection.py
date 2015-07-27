import pymysql as sql
import numpy as np

def getConnection():
    connection = sql.connect(host='thesisDBserver.dyn.adnovum.ch', user='vasilisp', passwd='a1b2c3$%^',db='OctoberLogsLocal')
    return connection

    
def connectAndFetch(query):
    connection = getConnection()
    
    cur = connection.cursor()    
    cur.execute(str(query))
    
    return cur.fetchall()

    
def getCursor():
    conn = getConnection()
    
    cur = conn.cursor()
    
    return cur
    
def connectAndFetchArray(query):
    connection = getConnection()
    
    cur = connection.cursor()    
    cur.execute(str(query))
    result = np.asarray(cur.fetchall())
    
    connection.close()
    
    return result
