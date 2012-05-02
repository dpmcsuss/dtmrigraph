'''
Created on Mar 12, 2012

@author: dsussman
'''
import pyximport;
pyximport.install()

import os
import numpy as np
from scipy.io import matlab
from scipy import sparse as sp 
import roi
import fibergraph
import zindex
import Embed
from matplotlib import pyplot as plt
from mayavi import mlab
from scipy.io import loadmat, savemat
from collections import Counter
from mayavi import mlab


def load_data(fn):
    m=matlab.loadmat(fn)
    g_all=m['fibergraph']
    n_all = np.shape(g_all)[0]
    non_iso = g_all.dot(np.ones(n_all)).nonzero()[0]
    g = g_all[non_iso[:,np.newaxis], non_iso]
    
    ncc,cc_label = sp.cs_graph_components(g)
    
    cc_size = [(cc,np.sum(cc_label==cc)) for cc in set(cc_label)]
    lcc = cc_size[np.argmax([cc[1] for cc in cc_size])]
    lcc_idx = np.nonzero((cc_label==lcc[1]))[0]
    
    return g, lcc_idx
    
    
def load_fibergraph(roi_fn, mat_fn):
    
    roix = roi.ROIXML(roi_fn+'.xml')
    rois = roi.ROIData(roi_fn+'.raw', roix.getShape())
    
    fg = fibergraph.FiberGraph(roix.getShape(),rois,[])
    fg.loadFromMatlab('fibergraph', '/mnt/braingraph1data/MRCAPgraphs/biggraphs/M87102217_fiber.mat')
    
    return fg
    
    
    

def get_roi(fg):
    nvertex = fg.spcscmat.shape[0]
    roival = np.array([fg.rois.get(zindex.MortonXYZ(v))  for v in xrange(nvertex)])
    
    inroi = np.nonzero(roival)[0]
    
    plt.hist(roival[inroi])
    G = fg.spcscmat[inroi,:][:,inroi]
    
    return G,inroi,roival

    
def embed(G, d):
    eA = Embed.Embed(d, Embed.self_matrix)

    eA.embed(G)
    return eA.get_scaled(d)
    
def get_lcc_idx(G):
    ncc,vertexCC = sp.cs_graph_components(G)
        
    cc_size = Counter(vertexCC)
    cc_size = sorted(cc_size.iteritems(), key=lambda cc: cc[1],reverse=True)
    cc_badLabel,_ = zip(*cc_size)
    cc_dict = dict(zip(cc_badLabel, np.arange(ncc+1)))
    
    vertexCC = [cc_dict[vcc] for vcc in vertexCC]
    
    
#    cc_size = [(cc,np.sum(cc_label==cc)) for cc in set(cc_label) if cc>0]
#    lcc = cc_size[np.argmax([cc[1] for cc in cc_size])][0]
#    lcc_idx = np.nonzero((cc_label==lcc))[0]
#    
    return np.array(vertexCC)
    
def save_lcc(fg, fn):
    vcc = get_lcc_idx(fg.spcscmat)
    
    np.save(open(fn+'_concomp.npy','w'),vcc)
    
    savemat(fn+'_concomp.mat',{'vertexCC':vcc})
    
    return vcc
    
def cc_for_each_brain(fiberDir, roiDir, ccDir, figDir):

    fiberSfx = '_fiber.mat'
    roiSfx = '_roi'
    
    brainFiles = [fn.split('_')[0] for fn in os.listdir(fiberDir)]
    
    for brainFn in brainFiles:
        print "Processing brain "+brainFn
        fg = load_fibergraph(roiDir+brainFn+roiSfx,fiberDir+brainFn+fiberSfx)
                                   
        vcc = save_lcc(fg, ccDir+brainFn)
        
        if figDir:
            save_figures(get_cc_coords(vcc,10), figDir+brainFn)
        
        del fg
        
def get_cc_coords(vcc, ncc):
    inlcc = (np.less_equal(vcc,ncc)*np.greater(vcc,0)).nonzero()[0]
    coord = np.array([zindex.MortonXYZ(v) for v in inlcc])

    return np.concatenate((coord,vcc[inlcc][np.newaxis].T),axis=1)
    
def save_figures(coord, fn):
    x,y,z,c = np.hsplit(coord,4)
    
    f = mlab.figure()
    mlab.points3d(x,y,z,c, mask_points=50, scale_mode='none',scale_factor=2.0)
    mlab.view(0,180)
    mlab.savefig(fn+'_view0,180.png',figure=f,magnification=4)
    mlab.view(0,90)
    mlab.savefig(fn+'_view0,90.png',figure=f,magnification=4)
    mlab.view(90,90)
    mlab.savefig(fn+'_view90,90.png',figure=f,magnification=4)
    mlab.close(f)
    
    
    
    
    
    
if __name__=='__main__':
    fiberDir = '/mnt/braingraph1data/MRCAPgraphs/biggraphs/'
    roiDir = '/mnt/braingraph1data/MR.new/roi/'
    ccDir = '/data/biggraphs/connectedcomp/'
    figDir = '/home/dsussman/Dropbox/Figures/DTMRI/lccPics/'

    brainfiles = [fn.split('_')[0] for fn in os.listdir(fiberDir)]
    
    
    
    
    roixml = '/mnt/braingraph1data/MR.new/roi/M87129789_roi.xml'
    roiraw = '/mnt/braingraph1data/MR.new/roi/M87129789_roi.raw'
    roifn = '/mnt/braingraph1data/MR.new/roi/M87129789_roi'
    matfn = '/mnt/braingraph1data/MRCAPgraphs/biggraphs/M87102217_fiber.mat'
    
    fg = load_fibergraph(roifn, matfn)