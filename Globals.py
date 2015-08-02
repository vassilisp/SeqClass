
import os,errno
import time

def mkdir_LR(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Creating PATH:', path)
    return path + '/'
    
def mkdir_RR(folder):
    directory = getResultsPATH() + folder
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('<<<<<<<><> CREATING PATH:', directory)
    return directory + '/'


def getResultsPATH():
    folder = "RESULTS"
    
    dirname = os.path.dirname
    
    
    directory = os.path.join(dirname(dirname(__file__)), folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory + '/'
    
def getProcessIDPath(processID, exeTime):
    path = processID + '_'+ exeTime + '/'
    full_path = mkdir_RR(path)
    
    return full_path
    