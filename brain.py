'''
Created on Mar 12, 2012

@author: dsussman
'''

import os
import numpy as np
from scipy import sparse as sp 
from scipy.io import loadmat, savemat

from matplotlib import pyplot as plt
from mayavi import mlab
from collections import Counter
from mayavi import mlab
    
import roi
import fibergraph
import zindex
import Embed
import lcc
import adjacency

from stfpSim import fastaprnn as FNN


graphDir = '/mnt/braingraph1data/projects/MRN/graphs/biggraphs/'
roiDir = '/mnt/braingraph1data/projects/MRN/base/roi/'
ccDir = '/data/biggraphs/connectedcomp/'
figDir = '/home/dsussman/Dropbox/Figures/DTMRI/lccPics/'
faDir = '/mnt/braingraph1data/projects/MRN/base/fa/'
mprDir = '/mnt/braingraph1data/projects/MRN/base/input/mprage_ss_crop/'
embDir = '/data/biggraphs/embedding/'
brainFiles = np.sort([fn.split('_')[0] for fn in os.listdir(ccDir)])
bfn = brainFiles[0]

mrnDict =  {"brainfiles":brainFiles,
            "roiDir":roiDir,
            "graphDir":graphDir,
            "embedDir":embDir,
            "lccDir":ccDir,
}



blsaDir = '/mnt/braingraph1data/projects/BLSA/aug12-mrcap/'
blsaBigGraph = blsaDir+'graphs/big/'
blsaROI = blsaDir+'intermediate/roi/'
blsaFiles = np.sort([fn.split('_')[0] for fn in os.listdir(blsaBigGraph)])
blsaLCC = '/data/projects/BLSA/graphs/lcc/'
blsaEmb = '/data/projects/BLSA/graphs/embed/'

blsaDict = {"brainfiles":blsaFiles,
            "roiDir":blsaROI,
            "graphDir":blsaBigGraph,
            "embedDir":blsaEmb,
            "lccDir":blsaLCC,
}
    
def load_fibergraph(roiDir, graphDir, bfn):
    
    roix = roi.ROIXML(roiDir+bfn+'_roi.xml')
    rois = roi.ROIData(roiDir+bfn+'_roi.raw', roix.getShape())
    
    fg = fibergraph.FiberGraph(roix.getShape(),rois,[])
    fg.loadFromMatlab('fibergraph', graphDir+bfn+'_fiber.mat')
    
    return fg
    
    

def get_roi(fg):
    nvertex = fg.spcscmat.shape[0]
    roival = np.array([fg.rois.get(zindex.MortonXYZ(v))  for v in xrange(nvertex)])
    
    inroi = np.nonzero(roival)[0]
    
    plt.hist(roival[inroi])
    G = fg.spcscmat[inroi,:][:,inroi]
    
    return G,inroi,roival


def get_emb_from_fn(roiDir, graphDir, ccDir, embedDir, bfn,e):
    print bfn
    print "Loading LCC"
    vcc = lcc.ConnectedComponent(fn=ccDir+bfn+'_concomp.npy')
    print "Loading Graph"
    fg = load_fibergraph(roiDir, graphDir, bfn)
    print "Get Induced Subgraph, Binarize, Symetrize"
    
    print "Embed"
    X = e.embed(G).get_scaled()

    if embedDir is not None:
        np.save(embedDir+bfn+'_embed.npy',X)
    return X


    
    
def get_3d(x,shape):
        """Takes a shape which is the shape of the new 3d image and 'colors' the image by connected component
        
        Input
        =====
        shape -- 3-tuple
        
        Output
        ======
        cc3d -- array of with shape=shape. colored so that ccz[x,y,z]=vcc[i] where x,y,z is the XYZ coordinates for Morton index i
        """
    
        cc3d = np.NaN*np.zeros(shape)
        allCoord = itt.product(*[xrange(sz) for sz in shape])
        
        [cc3d.itemset((xyz), x[zindex.XYZMorton(xyz)])
            for xyz in allCoord if not x[zindex.XYZMorton(xyz)]==0];
        return cc3d
    
def get_subgraph_density(A, v):
        return sum(sum(A[:,v][v,:]))/(len(v)**2)

def get_embedding_dim(G, dmax):
    n = G.shape[0]
    
    m = int(np.sqrt(n))
    
    rhoBS = np.array([get_subgraph_density(G,np.random.random_integers(0,n-1, m)) for _ in xrange(1000)])
    rhoHat = np.sort(.5-np.abs(.5-rhoBS))[950]
    threshold = 2*np.sqrt(rhoHat*(1-rhoHat)*n)
    
    svec,sval,_ = sp.linalg.svds(G,dmax)
    
    return np.sum(sval>threshold)

#def get_dim_for_scan(G,G2, v):
#    neigh = G2[:,v].nonzero()[0]
#    scan = G[:,neigh][neigh,:].toarray()
#    n = scan.shape[0]
#    return get_embedding_dim(scan,np.min((n-1,50)))
    
    
def get_dim_for_scan(G, v):
    neigh = G.neighbors(G.neighbors(v))
    scan = G.Adj[:,neigh][neigh,:].toarray()
    n = scan.shape[0]
    return get_embedding_dim(scan,np.min((n-1,50)))



def get_stfp_data_from_fn(roiDir, lccDir, embedDir, graphDir,bfn):
    """Get the embedding, roi class labels, and graph corresponding to the largest connected component"""
    fg = load_fibergraph(roiDir, graphDir, bfn)
    vcc = lcc.ConnectedComponent(fn=lccDir+bfn+'_concomp.npy')

    G = vcc.induced_subgraph(fg.spcscmat)
    X = np.load(embedDir+bfn+'_embedding.npy')
    
    inccIdx = (vcc.vertexCC==1).nonzero()[0]
    ccRoi = np.array([fg.rois.get(zindex.MortonXYZ(v)) for v in inccIdx])

    return X, ccRoi, G


if __name__=='__main__':


    brainfiles = [fn.split('_')[0] for fn in os.listdir(fiberDir)]
    
    
    
    
    roixml = '/mnt/braingraph1data/MR.new/roi/M87129789_roi.xml'
    roiraw = '/mnt/braingraph1data/MR.new/roi/M87129789_roi.raw'
    roifn = '/mnt/braingraph1data/MR.new/roi/M87129789_roi'
    matfn = '/mnt/braingraph1data/MRCAPgraphs/biggraphs/M87102217_fiber.mat'
    
    fg = load_fibergraph(roifn, matfn)