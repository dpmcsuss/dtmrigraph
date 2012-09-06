from stfpSim.fastaprnn import FAKNeighborsClassifier as FNN
from stfpSim.fastaprnn import FAKPseudoNeighbor as FPNN
from sklearn import cross_validation as cv
import numpy as np
from matplotlib import pyplot as plt
from random import sample
import time
import brain


def knn_experiments(brainfiles, kRange, dRange,fracRange, nfolds,leftRight=False, savefn=None):
    nrec = len(brainfiles)*len(kRange)*len(fracRange)*len(dRange)
    res = np.recarray(nrec,dtype=[('bfn',object),('k',int),('d',int),('frac',float),('Lhat',type(np.array))])
    idx = 0
    for bfn in brainfiles:
        x = np.load('/data/biggraphs/embedding/'+bfn+'_embedding.npy')
        y = np.load('/data/biggraphs/lccroi/'+bfn+'_lccroi.npy')
        if leftRight:
            y = np.greater(y,99).astype(int)
        for k in kRange:
            fnn = FNN(k)
            for d in dRange:
                for frac in fracRange:
                    lfo = cv.ShuffleSplit(x.shape[0], nfolds, frac)
                    Lhat = 1-cv.cross_val_score(fnn, x[:,:d],y,cv=lfo)
                    res[idx] = (bfn, k,d,frac,Lhat)
                    idx += 1
                    
        if savefn is not None:
            np.save(savefn, res)
    return res

def knn_experiments_no_clique(brainfiles, embedDir, graphDir, lccDir, roiDir,
                              kRange, dRange,trainFrac, testSize, nmc,leftRight=False,
                              savefn=None):
    nrec = len(brainfiles)*len(kRange)*len(trainFrac)*len(dRange)*nmc
    res = np.recarray(nrec,dtype=[('bfn',object),('k',int),('d',int),('frac',float),('Lhat',type(np.array))])
    idx = 0
    for bfn in brainfiles:
        print "Testing Brain: "+bfn
        startTime = time.time()
        x, y, G = brain.get_stfp_data_from_fn(roiDir, lccDir, embedDir, graphDir,bfn)
        
        if leftRight:
            y = np.greater(y,99).astype(int)
        
        for d in dRange:
            print "d="+repr(d)
            xd = x[:,:d]
            for frac in trainFrac:
                for mc  in np.arange(nmc):
                    trainSz = int(np.floor(frac*x.shape[0]))
                    train = np.array(sample(np.arange(x.shape[0]),trainSz))
                    test = np.array(sample(np.arange(trainSz), testSize)) # this is ok since we will ignore this when we test
                    fpnn = train_fpnn(xd[train,:],y[train],G[train,:],np.max(kRange))
                    for k in kRange:
                        Lhat = test_fpnn(fpnn,xd[train[test]],y[train[test]],k,test)
                        res[idx] = (bfn, k,d,frac,Lhat)
                        idx += 1
                    
        if savefn is not None:
            np.save(savefn, res)

        print "Time = "+repr(time.time()-startTime)
    return res



def train_fpnn(X, roi, G, kmax):
    fnn = FPNN(kmax)
    fnn.fit(X,roi, G)
    return fnn

def test_fpnn(fpnn, X, roi, k, testIdx):
    kMax = fpnn.n_neighbors
    fpnn.n_neighbors = k
    pred_labels = np.array([fpnn.predict(X[idx,:],testIdx[idx]) for idx in xrange(len(testIdx))]).flatten()
    Lhat = np.mean(np.not_equal(roi,pred_labels))

    return Lhat

    

def plot_err_vs_k(res):
    res = res[np.not_equal(res.bfn,None)]
    fracRange = np.unique(res.frac)
    kRange = np.unique(res.k)

    for f in fracRange:
        resf = [ np.concatenate(zip(get_match(res,k=k,frac=f).Lhat))
                 for k in kRange]
        mean,std = zip(*[(np.mean(r),np.std(r)) for r in resf])
        plt.errorbar(kRange, mean,std)

    plt.legend(['Test fraction = %.2f'%f for f in fracRange],loc='best')
        
def get_match(res,k=None, frac=None, d=None, returnLogical=False):
    n = len(res)
    kMatch = np.ones(n) if k is None else np.in1d(res.k,np.asarray(k))
    dMatch = np.ones(n) if d is None else np.in1d(res.d,np.asarray(d))
    fMatch = np.ones(n) if frac is None else np.in1d(res.frac,np.asarray(frac))
    
    match = np.logical_and(kMatch,np.logical_and(dMatch, fMatch))
    if returnLogical:
        return match
    else:
        return res[match]
    

def nikos_params():
    kRange = np.arange(1,28,2)
    fracRange = [.05,.1,.2]
    dRange = [50]
    nfolds = 1
    leftRight = True
    
    return kRange, dRange, fracRange, nfolds, leftRight

    
