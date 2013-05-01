from brain import *

import numpy as np
from sklearn import cross_validation
from scipy.ndimage import rotate


def get_dist_to_center(vcc):
    vccCoord = vcc.get_coords_for_lccs(1)
    vccCenter = np.mean(vccCoord, 0)
    distToVccCenter = np.sqrt(np.sum(np.pow(vccCoord - vccCenter, 2), 1))
    return distToVccCenter


def gaussian_average(x, y, mu, sig, getSe=True):
    p = norm.pdf(x, mu, sig)

#   tau = np.sum(p)/(x.shape[0]/10)
#   y = y[p>tau]
#   p = p[p>tau]

    mean = np.average(y, weights=p)
    if getSe:
        var = np.average((y - mean) ** 2, weights=p)
        df = (np.sum(p) ** 2) / np.sum(p ** 2)
        se = var / np.sqrt(df)
        return mean, se
    else:
        return mean


def smoothed_error_rate(x, err, xRange, bw):
    combined = zip(*[gaussian_average(x, err, xx, bw) for xx in xRange])
    return np.array(combined[0]), np.array(combined[1])


def smooth_err_rate_cv(x, err, xRange, bw):
    """Smooth with variable bandwidth selected via cross-val"""
    n = x.shape[0]
    n_iter = 5

    cv = cross_validation.ShuffleSplit(
        n, n_iterations=n_iter, test_fraction=.1, indices=True)

    k = 0

    for train_test in cv:
        # bwMin = bw[0] * np.ones_like(xRange)
        # bwMax = bw[1] * np.ones_like(xRange)

        # testX = 1

        while k < 5:
            fMin = np.array([gaussian_average(
                x, err, xx, bw, False) for xx, bw in xRange])
            fMax = np.array([gaussian_average(
                x, err, xx, bw, False) for xx in xRange])


def plot_smoothErr(x, y, xRange, bwRange, c=None):
    if c is None:
        c = dict(zip(bwRange,
                 [(512.0 - k) / 512.0 * np.array(plt.cm.jet(k))
                  for k in np.linspace(0, 255, len(bwRange)).astype(int)]))
        customC = False
    else:
        customC = True

    for i, bw in enumerate(bwRange):
        smoothErr, se = smoothed_error_rate(x, y, xRange, bw)

        if customC:
            cbw = c[i]
        else:
            cbw = c[bw]

        print repr(cbw)

        plt.fill_between(xRange, smoothErr + se,
                         smoothErr - se, color=cbw * .7)
        plt.plot(xRange, smoothErr, color=cbw,
                 linewidth=2, label=r'$\sigma={}$'.format(bw))


def get_3d_version(y, vcc, imgShape):
    inccIdx = (vcc.vertexCC == 1).nonzero()[0]
    oldCC = vcc.vertexCC

    roi = np.zeros_like(vcc.vertexCC, dtype=y.dtype)
    roi[inccIdx] = y

    vcc.vertexCC = roi
    roi3d = vcc.get_3d_cc(imgShape)
    vcc.vertexCC = oldCC

    return roi3d


def sideBysideRoi(vcc, y, yN, yC, s=None):

    imgShape = (149, 185, 149)

    y3d = get_3d_version(y, vcc, imgShape)

    yN3d = get_3d_version(yN, vcc, imgShape)

    yC3d = get_3d_version(yC, vcc, imgShape)

    if s is not None:

        plt.subplot(131)
        plt.imshow(yC3d[:, :, s])
        plt.subplot(132)
        plt.imshow(y3d[:, :, s])
        plt.subplot(133)
        plt.imshow(yC3d[:, :, s])
        plt.tight_layout()

    vcc.vertexCC = oldCC

    return y3d, yN3d, yC3d


def get_roi_boundary(roi3d, lccCoord, keepNan=False):
    neigh = np.array([[1, 0, 0],
                      [-1, 0, 0],
                      [0, 1, 0],
                      [0, -1, 0],
                      [0, 0, 1],
                      [0, 0, -1]])

    if keepNan:
        return np.array(
            [np.any([roi3d.item(*coord) != roi3d.item(*(coord + n))
                     for n in neigh]) for coord in lccCoord])
    else:
        return np.array(
            [np.any([roi3d.item(*coord) != roi3d.item(*(coord + n))
             and not np.isnan(roi3d.item(*(coord + n)))
             for n in neigh]) for coord in lccCoord])


def is_boundary(roi3d, coord):
    neigh = np.array([[1, 0, 0],
                      [-1, 0, 0],
                      [0, 1, 0],
                      [0, -1, 0],
                      [0, 0, 1],
                      [0, 0, -1]])

    return np.any([roi3d.item(*coord) != roi3d.item(*(coord + n))
                   for n in neigh])


def dist_to_boundary(distItem, coord):
    if distItem(*coord) != -1:
        return distItem(tuple(coord))

    neigh = np.array([[1, 0, 0],
                      [-1, 0, 0],
                      [0, 1, 0],
                      [0, -1, 0],
                      [0, 0, 1],
                      [0, 0, -1]])
    return np.nanmin([distItem(*(coord + n)) + np.linalg.norm(n)
                      for n in neigh])


def get_local(y3d, coord, k, keepNan=False):
    """Returns the k-neighborhood of the the point coord in y3d.

    Removes NaNs and defaults"""

    x, y, z = tuple(coord)
    nx, ny, nz = y3d.shape

    xl = max(x - k, 0)
    xu = min(x + k, nx - 1)
    yl = max(y - k, 0)
    yu = min(y + k, ny - 1)
    zl = max(z - k, 0)
    zu = min(z + k, nz - 1)

    yLocal = y3d[xl:xu, yl:yu, zl:zu]

    if not keepNan:
        yLocal = yLocal[~np.isnan(yLocal)]

    return yLocal


def get_local_chance(roi3d, coord, k):
    roiLocal = get_local(roi3d, coord, k)
    label = np.unique(roiLocal)
    try:
        return 1 - np.max([np.mean(roiLocal == l) for l in label])
    except:
        print repr(len(roiLocal))
        return 0


def get_local_error(y3d, yHat3d, coord, k):
    yLocal = get_local(y3d, coord, k)
    yHatLocal = get_local(yHat3d, coord, k)

    return np.mean(np.not_equal(yLocal, yHatLocal))


def randomize_boundary(voxelNeigh, y, p):
    yNew = np.array([voxNeigh[np.random.random_integers(len(voxNeigh)) - 1]
                     if voxNeigh.size > 0 else yy
                     for voxNeigh, yy in zip(voxelNeigh, y)])
    keepOld = np.random.random(y.shape) > p
    yNew = (yNew * (1 - keepOld)) + y * keepOld

    return yNew


def test_random_boundary(voxelNeigh, y, p, neighN, mask=None):
    yNew = randomize_boundary(voxelNeigh, y, p)

    if mask is None:
        return 1 - np.mean(yNew == yNew[neighN])
    else:
        return 1 - np.mean(yNew[mask] == yNew[neighN][mask])


def rotate_1vox(y3d, y, lccCoord, center):
    yNew = np.zeros_like(y)

    yItem = y3d.item

    for i, cc in enumerate(lccCoord):
        right = cc[0] > center[0]
        top = cc[2] > center[2]

        if right and top:
            cc = cc + np.array([0, 0, -1])
        elif not right and top:
            cc = cc + np.array([1, 0, 0])
        elif right and not top:
            cc = cc + np.array([-1, 0, 0])
        else:  # not right and not top
            cc = cc + np.array([0, 0, -1])

        if np.isnan(yItem(tuple(cc))):
            yNew[i] = y[i]

        else:
            yNew[i] = yItem(tuple(cc))

    return yNew


def rotate_degrees(y3d, lccCoord, d, axes=(0, 2), return3d=False):
    yNew3d = rotate(y3d, angle=d, axes=axes, order=0, reshape=False)
    #mask = ~np.isnan(y3d)
    yNewItem = yNew3d.item
    yItem = y3d.item
    yNew = np.array([yNewItem(tuple(c)) if not np.isnan(yNewItem(tuple(c)))
                     else yItem(tuple(c)) for c in lccCoord])
    if return3d:
        return yNew, yNew3d
    else:
        return yNew


def test_rotate_degrees(y3d, lccCoord, d, neigh, bm=True):
    yNew, yNew3d = rotate_degrees(y3d, lccCoord, d, return3d=True)

    labels = np.unique(yNew)

    chance = 1 - np.max([np.mean(yNew == l) for l in labels])

    err = 1 - np.mean(yNew == yNew[neigh])

    if bm:
        bm = get_roi_boundary(yNew3d, lccCoord)
        errBM = 1 - np.mean(yNew[bm] == yNew[neigh][bm])

        return err, chance, errBM

    else:
        return err, chance


def rotate_matrix(axis, theta):
    r = np.eye(3)
    a0 = axis[0]
    a1 = axis[1]

    r[a0, a0] = np.cos(theta)
    r[a0, a1] = -np.sin(theta)
    r[a1, a0] = np.sin(theta)
    r[a1, a1] = np.cos(theta)

    return r


def rotate_labels(y, y3d, lccCoord, theta, axis, center=None, return3d=False):
    if center is None:
        center = np.mean(lccCoord, 0)

    lccNew = np.round((lccCoord - center).dot(
        rotate_matrix(axis, theta)) + center).astype(int)

    y3dItem = y3d.item

    yNew = np.array([y3dItem(tuple(c)) for c in lccNew])
    # Anything left over we'll just keep the same
    yNew[np.isnan(yNew)] = y[np.isnan(yNew)]

    if not return3d:
        return yNew

    yNew3d = np.NaN * np.zeros_like(y3d, dtype=float)
    yNew3dSet = yNew3d.itemset

    # yNew3d[np.isnan(y3d)]=np.nan

    [yNew3dSet(tuple(cc), yy) for cc, yy in zip(lccCoord, yNew)]

    return yNew, yNew3d


def test_label(y, neighN, mask=None, returnChance=True, returnFull=True):
    chance = 1 - np.max([np.mean(y == l) for l in np.unique(y)])
    err = 1 - np.mean(y == y[neighN])

    if mask is not None:
        returnMask = True
        maskErr = 1 - np.mean(y[mask] == y[neighN][mask])
    else:
        returnMask = False
        maskErr = np.nan

    allRes = np.array([err, maskErr, chance])

    return allRes[np.array([returnFull, returnMask, returnChance])]


def rotate_labels_test_new_bm(y, y3d, lccCoord, neighN, theta, axis):
    yNew, yNew3d = rotate_labels(y, y3d, lccCoord, theta, axis, return3d=True)

    voxelNeigh = [get_local(yNew3d, coord, 1) for coord in lccCoord]
    bm = np.array([len(np.unique(v)) > 1 for v in voxelNeigh])

    print test_label(yNew, neighN, mask=bm, returnChance=False)

    return test_label(yNew, neighN, mask=bm, returnChance=False)
