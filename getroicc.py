import brain as br
import lcc
import numpy as np
import roi


def get_roi_cc(roixml, roiraw, ccfn, savefn):
    rois = roi.ROIData(roiraw, roi.ROIXML(roixml).getShape())
    vcc = lcc.ConnectedComponent(fn=ccfn)
    lcc3d = vcc.get_coords_for_lccs(1)[:, :3]
    vroi = np.array([rois.data[tuple(v3d)] for v3d in lcc3d.tolist()])
    np.save(savefn, vroi)
    return vroi


def _get_roi_cc(bfn):
    print 'Processing: ' + bfn

    roixml = br.roiDir + bfn + '_roi.xml'
    roiraw = br.roiDir + bfn + '_roi.raw'
    ccfn = br.ccDir + bfn + '_concomp.npy'
    savefn = '/data/biggraphs/lccroi/' + bfn + '_lccroi.npy'
    get_roi_cc(roixml, roiraw, ccfn, savefn)

if __name__ == '__main__':
    [_get_roi_cc(bfn) for bfn in br.brainFiles]
