
import os,errno

def mkdir_p(folder):
    directory = getResultsPATH() + folder
    if not os.path.exists(directory):
        os.makedirs(directory)


def getResultsPATH():
    folder = "RESULTS"
    
    dirname = os.path.dirname
    
    
    directory = os.path.join(dirname(dirname(__file__)), folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory + '/'
    
def getProcessIDPath(processID):
    return getResultsPATH() + processID + '/'
    