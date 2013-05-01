from fastaprnn import FAKNeighborsClassifier as FNN
from fastaprnn import FAKPseudoNeighbor as FPNN
from sklearn import cross_validation as cv
import numpy as np
from matplotlib import pyplot as plt
from random import sample

from scipy import sparse
from itertools import izip

import time
import brain

from itertools import product


def knn_experiments(brainfiles, kRange, dRange, fracRange,
                    nfolds, leftRight=False, savefn=None):
    nrec = len(brainfiles) * len(kRange) * len(fracRange) * len(dRange)
    res = np.recarray(nrec, dtype=[('bfn', object), ('k', int), (
        'd', int), ('frac', float), ('Lhat', type(np.array))])
    idx = 0
    for bfn in brainfiles:
        x = np.load('/data/biggraphs/embedding/' + bfn + '_embedding.npy')
        y = np.load('/data/biggraphs/lccroi/' + bfn + '_lccroi.npy')
        if leftRight:
            y = np.greater(y, 99).astype(int)
        for k in kRange:
            fnn = FNN(k)
            for d in dRange:
                for frac in fracRange:
                    lfo = cv.ShuffleSplit(x.shape[0], nfolds, frac)
                    Lhat = 1 - cv.cross_val_score(fnn, x[:, :d], y, cv=lfo)
                    res[idx] = (bfn, k, d, frac, Lhat)
                    idx += 1

        if savefn is not None:
            np.save(savefn, res)
    return res


def knn_experiments_no_clique(brainfiles, embedDir, graphDir,
                              lccDir, roiDir, kRange, dRange,
                              trainFrac, testSize, nmc,
                              leftRight=False, savefn=None):

    nrec = len(brainfiles) * len(kRange) * len(trainFrac) * len(dRange) * nmc
    res = np.recarray(nrec, dtype=[('bfn', object), ('k', int), (
        'd', int), ('frac', float), ('Lhat', type(np.array))])
    idx = 0
    for bfn in brainfiles:
        print "Testing Brain: " + bfn
        startTime = time.time()
        x, y, G = brain.get_stfp_data_from_fn(
            roiDir, lccDir, embedDir, graphDir, bfn)

        # add self loops to make sure each vertex is its own neighbor
        G = G + sparse.dia_matrix((np.ones(G.shape[0]), 0), shape=G.shape)

        n = x.shape[0]

        # permute to avoid nonsense
        perm = np.random.permutation(n)
        x = x[perm, :]
        y = y[perm, :]
        G = G[:, perm][perm, :]

        if leftRight:
            y = np.greater(y, 99).astype(int)

        for frac in trainFrac:
            for mc in np.arange(nmc):
                trainSz = int(np.floor(frac * x.shape[0]))
                train = np.array(sample(np.arange(x.shape[0]), trainSz))
                test = np.array(sample(np.arange(trainSz), testSize))
                for d in dRange:

                    print "d=" + repr(d)
                    xd = x[:, :d]
                    fpnn = train_fpnn(xd[train, :], y[train], G[train, :],
                                      np.max(kRange))
                    for k in kRange:
                        Lhat = test_fpnn(fpnn, xd[train[
                                         test]], y[train[test]], k, test)
                        res[idx] = (bfn, k, d, frac, Lhat)
                        idx += 1

        if savefn is not None:
            np.save(savefn, res)

        print "Time = " + repr(time.time() - startTime)
    return res


def knn_experiments_compare(brainfiles, embedDir, graphDir,
                            lccDir, roiDir, kRange, dRange,
                            fracRange, nmc, testSize=None,
                            leftRight=False, savefn=None):
    """
    Compare the K-nearest-neighbors when training on graph
    neighbors or when ignoring neighbors

    @INPUT:
    brainFiles ---  the UIDs associated with each brain
    *Dir --- where to find the different stuff to load
    kRange --- range of nearest neighbors
    """
    nrec = len(kRange) * len(dRange)

    for bfn in brainfiles:
        print "Testing Brain: " + bfn
        startTime = time.time()
        x, y, G = brain.get_stfp_data_from_fn(
            roiDir, lccDir, embedDir, graphDir, bfn)

        if leftRight:
            y = np.greater(y, 99).astype(int)

        for frac in fracRange:
            lfo = cv.ShuffleSplit(x.shape[0], nmc, frac)

            for train, test in lfo:
                resC = np.recarray(nrec, dtype=[('bfn', object), ('k', int), (
                    'd', int), ('frac', float), ('Lhat', float), ('test')])
                resN = np.recarray(nrec, dtype=[('bfn', object), (
                    'k', int), ('d', int), ('frac', float), ('Lhat', float)])

                idx = 0
                print "Run " + repr(
                    idx + 1) + "for training fraction " + repr(frac)

                test = test.nonzero()[0]
                if testSize is not None:
                    test = np.array(sample(test, testSize))
                for d in dRange:
                    xd = x[:, :d]
                    fpnn = train_fpnn(xd[train, :], y[train],
                                      G[train, :], np.max(kRange))
                    fnn = train_fnn(xd[train, :], y[train], np.max(kRange))
                    for k in kRange:
                        Lhat = test_fpnn(fpnn, xd[test, :], y[test], k, test)
                        resN[idx] = (bfn, k, d, frac, Lhat)

                        Lhat = test_fnn(fnn, xd[test, :], y[test], k)
                        resC[idx] = (bfn, k, d, frac, Lhat)
                        idx += 1

                if savefn is not None:
                    try:
                        old = np.load(savefn[0])
                        np.save(savefn[0], np.concatenate((old, resC)))
                    except IOError:
                        print "Saving new file"
                        np.save(savefn[0], resC)
                    try:
                        old = np.load(savefn[1])
                        np.save(savefn[1], np.concatenate((old, resN)))
                    except IOError:
                        print "Saving new file"
                        np.save(savefn[1], resN)

        print "Time = " + repr(time.time() - startTime)


def all_whole_brain_results(bfiles, mrnDict, kRange, dRange,
                            nfolds, savefn, testSz):

    if savefn is None:
        print "WARNING NOT SAVING"

    kMax = np.max(kRange)

    for bfn in bfiles:
        print "Loading data for brain {}".format(bfn)
        xF, y, G = brain.get_stfp_data_from_fn(bfn=bfn, **mrnDict)

        nnodes = xF.shape[0]

        p = np.random.permutation(nnodes)
        xF = xF[p, :]
        y = y[p, :]
        G = G[p, :][:, p]

        crossval = cv.StratifiedKFold(y, nfolds, indices=True)

        if testSz is not None:
            testSub = np.empty((nfolds, testSz), dtype=int)
            for (train, test), fold in izip(crossval, xrange(nfolds)):
                testSub[fold, :] = np.array(sample(test, testSz))

        for d in dRange:
            fold = 0
            x = xF[:, :d]
            fpnnList = np.empty(nfolds, dtype=object)

            print "Training for d={}...".format(d)
            for (train, test), fold in izip(crossval, xrange(nfolds)):
                fpnnList[fold] = train_fpnn(x[train, :], y[train],
                                            G[train, :], kMax)

            for k in kRange:
                print "Testing for k={} ...".format(k)

                yN = np.zeros_like(y)
                yC = np.zeros_like(y)
                LN = np.ones(nfolds)
                LC = np.ones(nfolds)

                for (train, test), fold in izip(crossval, xrange(nfolds)):
                    fpnn = fpnnList[fold]
                    fpnn.n_neighbors = k

                    if testSz is not None:
                        test = testSub[fold, :]

                    # Get Predicted Labels using No Cliques
                    yN[test] = np.array([fpnn.predict(x[idx, :], idx) for idx
                                         in test]).flatten().astype(yN.dtype)
                    LN[fold] = np.mean(np.not_equal(y[test], yN[test]))
                    print "No Clique Error = {}".format(LN[fold])

                    # and using cliques, ie no idx ............. v ... there
                    yC[test] = np.array([fpnn.predict(x[idx, :]) for idx
                                         in test]).flatten().astype(yN.dtype)
                    LC[fold] = np.mean(np.not_equal(y[test], yC[test]))

                    print "With Clique Error = {}".format(LC[fold])

                yN = yN[np.argsort(p)]
                yC = yC[np.argsort(p)]

                if savefn is not None:
                    np.savez(savefn + '{}_k={}_d={}_folds={}.npz'.format(
                        bfn, k, d, nfolds), yN, yC, LN, LC)

                # END for k
            # END for d


def knn_error_plot(bfiles, kRange, dRange, nfolds, savefn):
    nd = dRange.shape[0]
    nk = kRange.shape[0]
    nb = len(bfiles)

    c = [(10.0 - k) / 10.0 * np.array(plt.cm.jet(k * 50 + 30))
         for k in np.arange(len(kRange))]

    for k, j in zip(kRange, xrange(nk)):
        LN = np.zeros((nd, nfolds * nb))
        LC = np.zeros((nd, nfolds * nb))

        for d, i in zip(dRange, xrange(nd)):
            LN[i, :] = np.concatenate(
                [np.load(savefn + '{}_k={}_d={}_folds={}.npz'.
                         format(bfn, k, d, nfolds))['arr_2']
                 for bfn in bfiles])

            LC[i, :] = np.concatenate(
                [np.load(savefn + '{}_k={}_d={}_folds={}.npz'.
                         format(bfn, k, d, nfolds))['arr_3']
                 for bfn in bfiles])

        LCmean = np.mean(LC, 1)
        LNmean = np.mean(LN, 1)
        LCse = np.std(LC, 1) / np.sqrt(nb * nfolds)
        LNse = np.std(LN, 1) / np.sqrt(nb * nfolds)

        labelC = r'$\hat{L}$,' + ' $k={}$'.format(k)
        labelN = r'$\tilde{L}$,' + ' $k={}$'.format(k)

        plt.errorbar(dRange, LCmean, yerr=LCse, fmt='-',
                     color=c[j], linewidth=j + 2, label=labelC)
        plt.errorbar(dRange, LNmean, yerr=LNse, fmt='--',
                     color=c[j], linewidth=j + 2, label=labelN)


def get_cv_neighbors(x, y, G, crossval, k, d, permute=False):
    x = x[:, :d]
    # inccIdx = (vcc.vertexCC==1).nonzero()[0]

    nnodes = x.shape[0]

    if permute:
        # Permute everything
        print "Permuting Vertices"
        p = np.random.permutation(nnodes)
        x = x[p, :]
        y = y[p]
        G = G[p, :][:, p]

    neighC = np.zeros((nnodes, k), dtype=np.int32)
    neighN = np.zeros((nnodes, k), dtype=np.int32)

    if type(crossval) is int:
        crossval = cv.StratifiedKFold(y, crossval, indices=True)
    nfolds = crossval.k

    for fold, (train, test) in enumerate(crossval):
        print "Training fold", fold

        fpnn = train_fpnn(x[train, :], y[train], G[train, :], k)

        print "Getting Neighbors"

        neighC[test] = np.squeeze(
            [train[fpnn.kneighbors(x[idx, :], return_distance=False)]
             for idx in test]).reshape(neighC[test].shape)

        print "Getting Non-Graph-Neighbor Neighbors"

        neighN[test] = np.squeeze(
            [train[fpnn.kneighbors(x[idx, :], idx, return_distance=False)]
             for idx in test]).reshape(neighN[test].shape)

    if permute:
        pinv = np.argsort(p)
        neighN = p[neighN[pinv, :]]
        neighC = p[neighC[pinv, :]]

    if k == 1:
        neighN = np.squeeze(neighN)
        neighC = np.squeeze(neighC)

    return neighN, neighC


def get_whole_brain_results(x, y, G, k, d, crossval, permute=False):

    # print "Loading Brain "+bfn
    # x, y, G = brain.get_stfp_data_from_fn(roiDir, lccDir, embedDir,
    # graphDir,bfn)
    print " "
    print "Performing KNN K-fold Stratified CV for k={} and d={}.".format(k, d)

    x = x[:, :d]
    # inccIdx = (vcc.vertexCC==1).nonzero()[0]

    nnodes = x.shape[0]

    if permute:
        # Permute everything
        print "Permuting Vertices"
        p = np.random.permutation(nnodes)
        x = x[p, :]
        y = y[p, :]
        G = G[p, :][:, p]

    if type(crossval) is int:
        crossval = cv.StratifiedKFold(y, crossval, indices=True)

    nfolds = crossval.k

    fold = 0
    yHatN = np.zeros_like(y)
    yHatC = np.zeros_like(y)
    LhatN = np.ones(nfolds)
    LhatC = np.ones(nfolds)

    for train, test in crossval:
        # print "Training for fold {}".format(fold+1)
        # Train our classifier
        fpnn = train_fpnn(x[train, :], y[train], G[train, :], k)
        # print "Done Training"

        # Get Predicted Labels using No Cliques
        yHatN[test] = np.array(
            [fpnn.predict(x[idx, :], idx)
             for idx in test]).flatten().astype(yHatN.dtype)
        LhatN[fold] = np.mean(np.not_equal(y[test], yHatN[test]))
        # print "No Clique Error = {}".format(LhatN[fold])

        # and using cliques, ie no idx ............. v ... there
        yHatC[test] = np.array(
            [fpnn.predict(x[idx, :])
             for idx in test]).flatten().astype(yHatN.dtype)
        LhatC[fold] = np.mean(np.not_equal(y[test], yHatC[test]))
        # print "With Clique Error = {}".format(LhatC[fold])

        fold += 1

    if permute:
        yHatN = yHatN[np.argsort(p)]
        yHatC = yHatC[np.argsort(p)]

    print "No Clique Error = {}".format(np.mean(LhatN))
    print "Clique Error = {}".format(np.mean(LhatC))

    return yHatN, yHatC, LhatN, LhatC


def train_fpnn(X, roi, G, kmax):
    fnn = FPNN(kmax)
    fnn.fit(X, roi, G)
    return fnn


def test_fpnn(fpnn, X, roi, k, testIdx):
    kMax = fpnn.n_neighbors
    fpnn.n_neighbors = k
    pred_labels = np.array(
        [fpnn.predict(X[idx, :], testIdx[idx])
         for idx in xrange(len(testIdx))]).flatten()
    Lhat = np.mean(np.not_equal(roi, pred_labels))

    fpnn.n_neighbors = kMax
    return Lhat


def train_fnn(X, roi, kmax):
    fnn = FNN(kmax)
    fnn.fit(X, roi)
    return fnn


def test_fnn(fnn, X, roi, k):
    kMax = fnn.n_neighbors
    fnn.n_neighbors = k
    Lhat = 1 - fnn.score(X, roi)
    fnn.n_neighbors = kMax
    return Lhat


def plot_err_vs_k(res):
    res = res[np.not_equal(res.bfn, None)]
    fracRange = np.unique(res.frac)
    kRange = np.unique(res.k)

    for f in fracRange:
        resf = [np.concatenate(zip(get_match(res, k=k, frac=f).Lhat))
                for k in kRange]
        mean, std = zip(*[(np.mean(r), np.std(r)) for r in resf])
        plt.errorbar(kRange, mean, std)

    plt.legend(['Test fraction = %.2f' % f for f in fracRange], loc='best')


def get_match(res, k=None, frac=None, d=None, returnLogical=False):
    n = len(res)
    kMatch = np.ones(n) if k is None else np.in1d(res.k, np.asarray(k))
    dMatch = np.ones(n) if d is None else np.in1d(res.d, np.asarray(d))
    fMatch = np.ones(n) if frac is None else np.in1d(
        res.frac, np.asarray(frac))

    match = np.logical_and(kMatch, np.logical_and(dMatch, fMatch))
    if returnLogical:
        return match
    else:
        return res[match]


def nikos_params():
    kRange = np.arange(1, 28, 2)
    fracRange = [.05, .1, .2]
    dRange = [50]
    nfolds = 1
    leftRight = True

    return kRange, dRange, fracRange, nfolds, leftRight


def get_3d_version(y, vcc, imgShape):
    inccIdx = (vcc.vertexCC == 1).nonzero()[0]
    oldCC = vcc.vertexCC
    vcc.vertexCC = np.zeros_like(oldCC)
    vcc.vertexCC[inccIdx] = y
    roi3d = vcc.get_3d_cc(imgShape)
    vcc.vertexCC = oldCC

    return roi3d


def sideBysideRoi(vcc, y, yhat, s=None):

    imgShape = (149, 185, 149)
    inccIdx = (vcc.vertexCC == 1).nonzero()[0]
    oldCC = vcc.vertexCC

    roi = np.zeros_like(vcc.vertexCC)
    roi[inccIdx] = y

    vcc.vertexCC = roi
    roi3d = vcc.get_3d_cc(imgShape)

    roiHat = np.zeros_like(vcc.vertexCC)
    roiHat[inccIdx] = yhat
    vcc.vertexCC = roiHat
    roiHat3d = vcc.get_3d_cc(imgShape)

    vcc.vertexCC = oldCC

    if s is not None:

        plt.subplot(121)
        plt.imshow(roi3d[:, :, s])
        plt.subplot(122)
        plt.imshow(roiHat3d[:, :, s])
        plt.tight_layout()

    vcc.vertexCC = oldCC

    return roi3d, roiHat3d
