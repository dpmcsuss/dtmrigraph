
import zindex
import numpy as np
from itertools import product
from matplotlib import pyplot as plt


def get_3d(x, shape):
    """Takes a shape which is the shape of the new 3d image and 'colors'
    the image by connected component
    def
    Input
    =====
    shape -- 3-tuple

    Output
    ======
    cc3d -- array of with shape=shape. colored so that ccz[x,y,z]=vcc[i]
    where x,y,z is the XYZ coordinates for Morton index i
    """

    cc3d = np.NaN * np.zeros(shape)
    allCoord = product(*[xrange(sz) for sz in shape])

    [cc3d.itemset((xyz), x[zindex.XYZMorton(xyz)])
        for xyz in allCoord if not x[zindex.XYZMorton(xyz)] == 0]
    return cc3d


def get_roi_subgraph(fg):
    """Takes a fibergraph object and returns the ROI info and the induced
    subgraph

    Input
    =======
    fg  -- a fibergraph object which has ROI data

    Output
    =======
    G -- ROI induced subgraph
    inroi -- Indicator vector of whether ROIs are in voxel or not
    roival --  the roi values for each vertex in the induced subgraph
    """

    nvertex = fg.spcscmat.shape[0]
    roival = np.array(
        [fg.rois.get(zindex.MortonXYZ(v)) for v in xrange(nvertex)])

    inroi = np.nonzero(roival)[0]

    plt.hist(roival[inroi])
    G = fg.spcscmat[inroi, :][:, inroi]

    return G, inroi, roival
