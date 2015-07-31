class Queries:
    
    def getTransPerDayQ(processID):
        query = "SELECT from_unixtime(timestamp div 1000, '%d.%m') AS dateID , count(refererID) FROM Preprocess  WHERE processID = ':processID' GROUP BY dateID ORDER BY timestamp ASC, clientId ASC"
        return query.replace(":processID", processID,1)
        
    def getTransPerUserQ(processID):
        query = "SELECT userId, count(refererId) as cnt FROM `Preprocess` where processID = ':processID' GROUP BY userId order by cnt"
        return query.replace(":processID", processID,1)
        
    def getTransPerClientIdQ(processID):
        query = "SELECT userId, clientId, count(refererID) AS cnt FROM Preprocess WHERE processID = ':processID' GROUP BY clientId ORDER BY userId ASC, cnt DESC"
        return query.replace(":processID", processID,1)
        
    def getTransPerSubSessionQ(processID):
        query = "SELECT userId, concat(clientId,subSession) AS divisor, COUNT(refererID) AS cnt FROM Preprocess WHERE processID = ':processID' GROUP BY clientId, subSession"
        return query.replace(":processID", processID,1)

        
    def getTransFrequencies(processID):
        query = "SELECT concat(refererID, targetID) AS transi, count(refererID) AS cnt FROM Preprocess WHERE processID = ':processID' GROUP BY refererID,targetID ORDER BY cnt DESC "
        return query.replace(":processID", processID,1)
        
    def getPagesFrequencies(processID):
        query = """ CREATE TEMPORARY TABLE t1 SELECT refererID AS pages 
                    FROM Preprocess WHERE processID= ':processID';
                    INSERT t1 SELECT targetID FROM Preprocess;
                    SELECT pages, count(pages) AS cnt FROM t1 GROUP BY pages ORDER BY cnt DESC;
                """
        return query.replace(":processID", processID,1)
            
    def getTransPerDayPerUser(procesID):
        query = """ SELECT userId, from_unixtime(timestamp div 1000, '%m.%d') AS dateID ,
                        clientId, subSession ,count(refererID) AS cnt
                    FROM Preprocess where processID = ':processID'
                    GROUP BY userId, dateID, clientId, subSession
                    ORDER BY userId, dateID
                """
        return query.replace(":processID", processID,1)
        
    def getTransPerUserPerClientId(processID):
        query = """
                SELECT userId, clientId, count(refererID)
                FROM Preprocess where processID = ':processID'
                GROUP BY clientId
                ORDER BY userId
                """
        return query.replace(":processID", processID, 1)
 