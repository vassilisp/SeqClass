import numpy as np

def rebatch(X_train, X_test, Y_train, Y_test, batchN, pardon = None):
    
    
    X_train, Y_train = single_rebatcher(X_train, Y_train, batchN, pardon)
    X_test, Y_test = single_rebatcher(X_test, Y_test, batchN, pardon)
    
    return X_train, X_test, Y_train, Y_test
    
    
def single_rebatcher(X, Y, batchN, pardon = 0.25):
    cnt = 0
    buf = ''
    X_buf = []
    Y_buf = []
    
    #--check if we will use the clientId for divider (divider 0)
    if(batchN == 0):
        from reporter import Reporter
        report = Reporter()
        report.new_report('Rebatcher report')
        report.report('using full sequence (clientId)')
        return X,Y, report
    
    
    if pardon == 0 or pardon == None:
        pardon_up = 1
        pardon_down = 0
    else:
        pardon_up = (1 - pardon)
        pardon_down = pardon
        
    for i,seq in enumerate(X):
        A = seq.split()
        numTrans = len(A)
        
        batchInA = numTrans//batchN
        
        stop = 0
        for j in range(batchInA):
            start = j*batchN
            stop = (j+1) * batchN
            
            buf = A[start:stop]
            
            X_buf.append(' '.join(buf))
            Y_buf.append(Y[i])
        
        x_remaining = A[stop:]
        if (len(x_remaining) <= pardon_down * batchN) and (batchInA != 0):
            X_buf[-1] = X_buf[-1] + ' ' + ' '.join(x_remaining)
        elif len(x_remaining) >= pardon_up * batchN:
            X_buf.append(' '.join(x_remaining))
            Y_buf.append(Y[i])
            
    X_buf = np.asarray(X_buf)
    Y_buf = np.asarray(Y_buf)
    
    old_total=0
    for old,seq in enumerate(X):
        old_total += len(seq.split()) + 1

    new_total=0    
    for new,seq in enumerate(X_buf):
        new_total += len(seq.split()) + 1
        
    from reporter import Reporter
    report = Reporter()
    report.new_report('Rebatcher report')
    report.report('old sample --total transitions: ,' +  str(old_total) + '-- total sequences: ,' + str(old))
    report.report('new sample --total transitions: ,' +  str(new_total) + '-- total sequences: ,' + str(new))
    return X_buf, Y_buf, report

def getBatchStatistics(X, Y):
    pass


if __name__ == "__main__":
    print('running test sequence')
    test_x = ['1 2 3 4 5 6 7 8 9 0 1','10 11 12 13 14','20 21 21 22 23 24']
    test_y = [0,1,2]
    print(test_x, test_y)
    print('-'*30)
    print(single_rebatcher(test_x, test_y, 4))