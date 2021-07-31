# %% md

## Libraries

# %%

import cv2
import numpy as np
import pyrtools as pt
from scipy import signal
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt


# %% md

## Utils

# %%

def RGB2Lab(im):
    im = np.float32(im) / 255

    mask = im >= 0.04045
    im[mask] = ((im[mask] + 0.055) / 1.055) ** 2.4
    im[~mask] = im[~mask] / 12.92

    matrix = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])

    c_im = np.dot(im, matrix.T)
    c_im[:, :, 0] = c_im[:, :, 0] / 95.047
    c_im[:, :, 1] = c_im[:, :, 1] / 100.000
    c_im[:, :, 2] = c_im[:, :, 2] / 108.833

    mask = c_im >= 0.008856
    c_im[mask] = c_im[mask] ** (1 / 3)
    c_im[~mask] = 7.787 * c_im[~mask] + 16 / 116

    im_Lab = np.zeros_like(c_im)

    im_Lab[:, :, 0] = (116 * c_im[:, :, 1]) - 16
    im_Lab[:, :, 1] = 500 * (c_im[:, :, 0] - c_im[:, :, 1])
    im_Lab[:, :, 2] = 200 * (c_im[:, :, 1] - c_im[:, :, 2])

    return im_Lab


def normlize(arr):
    return ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')


def conv2(x, y, mode=None):
    if mode == 'same':
        return np.rot90(signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
    else:
        return signal.convolve2d(x, y)


def RRoverlapconv(kernel, in_):
    out = conv2(in_, kernel, mode='same')
    rect = np.ones_like(in_)

    overlapsum = conv2(rect, kernel, 'same')
    out = np.sum(kernel) * out / overlapsum
    return out


def RRgaussfilter1D(halfsupport, sigma, center=0):
    t = list(range(-halfsupport, halfsupport + 1))
    kernel = np.array([np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) for x in t])
    kernel = kernel / sum(kernel)

    return kernel.reshape(1, kernel.shape[0])


def DoG1filter(a, sigma):
    sigi = 0.71 * sigma
    sigo = 1.14 * sigma

    t = range(-a, a + 1)

    gi = [np.exp(-x ** 2 / (2 * sigi ** 2)) for x in t]
    gi = gi / sum(gi)
    go = [np.exp(- x ** 2 / (2 * sigo ** 2)) for x in t]
    go = go / sum(go)

    return gi.reshape(1, gi.shape[0]), go.reshape(1, go.shape[0])


def addborder(im, xbdr, ybdr, arg):
    """
    imnew = addborder(im,xborder,yborder,arg)  Make image w/added border.
    imnew = addborder(im,5,5,128)  Add 5 wide border of val 128.
    imnew = addborder (im,5,5,'even')  Even reflection.
    imnew = addborder (im,5,5,'odd')  Odd reflection.
    imnew = addborder (im,5,5,'wrap')  Wraparound.
    """
    ysize, xsize = im.shape

    #     check thickness
    if (xbdr > xsize) or (ybdr > ysize):
        raise ValueError('borders must be thinner than image')

    #     if arg is a number, fill border with its value.
    if isinstance(arg, (int, float)):
        imbig = cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_CONSTANT, value=arg)

    #     Even reflections
    elif arg == 'even':
        imbig = cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_REFLECT)

    #     Odd reflections
    elif arg == 'odd':
        imbig = cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_REFLECT_101)

    #    Wraparound
    elif arg == 'wrap':
        imbig = cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_WRAP)
    else:
        raise ValueError('unknown border style')
    return imbig


def filt2(kernel, im1, reflect_style='odd'):
    """
    im2 = filt2(kernel,im1,reflect_style)
    Improved version of filter2 in MATLAB, which includes reflection.
    Default style is 'odd'. Also can be 'even', or 'wrap'.
    im2 = filt2(kern,image)  apply kernel with odd reflection (default).
    im2 = filt2(kern,image,'even')  Use even reflection.
    im2 = filt2(kern,image,128)  Fill with 128's.

    Ruth Rosenholtz
    """

    ky, kx = kernel.shape
    iy, ix = im1.shape

    # TODO: index should be checked, maybe it needs to be changed 1 pixel (getting back -1)
    imbig = addborder(im1, kx, ky, reflect_style)
    imbig = conv2(imbig, kernel, 'same')
    im2 = imbig[ky:ky + iy, kx:kx + ix]

    return im2


def RRcontrast1channel(pyr, DoG_sigma=2):
    levels = len(pyr)
    contrast = [0] * levels

    innerG1, outerG1 = DoG1filter(round(DoG_sigma * 3), DoG_sigma)

    for i in range(0, levels):
        inner = filt2(innerG1, pyr[(i, 0)])
        inner = filt2(innerG1.T, inner)
        outer = filt2(outerG1, pyr[(i, 0)])
        outer = filt2(outerG1.T, outer)
        tmp = inner - outer
        contrast[i] = abs(tmp)

    return contrast


def reduce(image0, kernel=None):
    """
    Reduce: for building Gaussian or Laplacian pyramids. 1-D separable kernels.

    imnew = reduce(im0) Reduce w/default kernel: [.05 .25 .4 .25 .05]
    imnew = reduce(im0, kern) Reduce with kern; sums to unity.

    Ruth Rosenholtz 
    """

    if kernel is None:
        #     Default kernel
        kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])

    ysize, xsize = image0.shape

    image0 = filt2(kernel, image0)  # Filter horizontally.
    #     filt2 is filter2 with reflection.
    image1 = image0[:, range(0, xsize, 2)]

    image1 = filt2(kernel.T, image1)  # Filter vertically.
    image2 = image1[range(0, ysize, 2), :]

    return image2


def RRoverlapconvexpand(in_, kernel=None):
    """
    out = RRoverlapconvexpand(in_)  return an image expanded to double size,
    out = RRoverlapconvexpand(in, kernel); specify 1-D kernel with unity sum.
    """

    if kernel is None:
        #     Default kernel
        kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])

    ysize, xsize = in_.shape
    kernel = kernel * 2  # kernel sum=2 to account for padding.

    tmp = np.zeros([ysize, 2 * xsize])  # First double the width
    k = list(range(0, xsize))
    k_2 = [x * 2 for x in k]
    tmp[:, k_2] = in_[:, k]
    tmp = RRoverlapconv(kernel, tmp)  # ..and filter horizontally.

    out = np.zeros([2 * ysize, 2 * xsize])  # Next double the height
    k = list(range(0, ysize))
    k_2 = [x * 2 for x in k]
    out[k_2, :] = tmp[k, :]
    out = RRoverlapconv(kernel.T, out)  # ..and filter vertically.

    return out


def HV(in_):
    out = in_[0] - in_[1]
    return out


def DD(in_):
    out = in_[3] - in_[2]
    return out


def sumorients(in_):
    out = in_[0] + in_[1] + in_[2] + in_[3]
    return out


def poolnew(in_, sigma=None):
    """
    Pools with a gaussian.  Note assumes that input image is actually
    4 equal-size images, side by side.

    Usage: out = poolnew(input_image, sigma)

    """

    in1 = in_[0]  # H -> first quarter
    in2 = in_[1]  # V -> second quarter
    in3 = in_[2]  # L -> third quarter
    in4 = in_[3]  # R -> last quarter

    if sigma is None:
        out1 = reduce(RRoverlapconvexpand(in1))
        out2 = reduce(RRoverlapconvexpand(in2))
        out3 = reduce(RRoverlapconvexpand(in3))
        out4 = reduce(RRoverlapconvexpand(in4))
    else:
        kernel = RRgaussfilter1D(round(2 * sigma), sigma)
        out1 = reduce(RRoverlapconvexpand(in1, kernel), kernel)
        out2 = reduce(RRoverlapconvexpand(in2, kernel), kernel)
        out3 = reduce(RRoverlapconvexpand(in3, kernel), kernel)
        out4 = reduce(RRoverlapconvexpand(in4, kernel), kernel)

    out = out1, out2, out3, out4

    return out


def imrotate(im, angle, method='nearest', bbox='crop'):
    func_method = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    func_bbox = {'loose': True, 'crop': False}
    PIL_im = Image.fromarray(im)

    im_rot = PIL_im.rotate(angle, expand=func_bbox[bbox], resample=func_method[method])
    return np.array(im_rot)


def imrotate2(im, angle, method='cubic', bbox='crop'):
    #     By default rotate uses cubic interpolation
    return ndimage.rotate(im, angle=angle)


def orient_filtnew(pyr, sigma=16 / 14):
    """
    ORIENT_FILTNEW Filters "pyr" (in principle, one level of the
        Gaussian pyramid generated by gausspyr) with 2nd
        derivative filters in 4 directions

    Usage: hvdd = orient_filt(pyr)
        Where hvdd is the 4 output images concatenated 
        together, in the order horizontal, vertical, up-left,
        and down-right.
    """

    halfsupport = round(3 * sigma)
    #     halfsupport was 10, for default sigma.  We need a halfsupport of about
    #     2*sigma for a single Gaussian.  Here we have three, one at -sigma, one at
    #     sigma, so we should need a halfsupport of about 3*sigma.

    sigy = sigma
    sigx = sigma  # Was sigx = 3*sigma.

    gx = RRgaussfilter1D(halfsupport, sigx)
    gy = RRgaussfilter1D(halfsupport, sigy, sigma)
    Ga = conv2(gx, gy.T)
    Ga = Ga / sum(sum(Ga))
    gy = RRgaussfilter1D(halfsupport, sigy)
    Gb = conv2(gx, gy.T)
    Gb = Gb / sum(sum(Gb))
    gy = RRgaussfilter1D(halfsupport, sigy, -sigma)
    Gc = conv2(gx, gy.T)
    Gc = Gc / sum(sum(Gc))
    H = -Ga + 2 * Gb - Gc
    V = H.T

    GGa = imrotate2(Ga, 45, 'bicubic', 'crop')
    GGa = GGa / sum(sum(GGa))
    GGb = imrotate2(Gb, 45, 'bicubic', 'crop')
    GGb = GGb / sum(sum(GGb))
    GGc = imrotate2(Gc, 45, 'bicubic', 'crop')
    GGc = GGc / sum(sum(GGc))
    R = -GGa + 2 * GGb - GGc
    GGa = imrotate2(Ga, -45, 'bicubic', 'crop')
    GGa = GGa / sum(sum(GGa))
    GGb = imrotate2(Gb, -45, 'bicubic', 'crop')
    GGb = GGb / sum(sum(GGb))
    GGc = imrotate2(Gc, -45, 'bicubic', 'crop')
    GGc = GGc / sum(sum(GGc))
    L = -GGa + 2 * GGb - GGc

    hout = filt2(H, pyr)
    vout = filt2(V, pyr)
    lout = filt2(L, pyr)
    rout = filt2(R, pyr)

    hvdd = hout, vout, lout, rout

    return hvdd


def entropy(x, nbins=None):
    nsamples = x.shape[0]

    if nbins is None:
        nbins = int(np.ceil(np.sqrt(nsamples)))

    ref_range = (x.min(), x.max())
    ref_hist, _ = np.histogram(x, bins=nbins, range=ref_range)

    ref_hist = ref_hist / float(np.sum(ref_hist))
    ref_hist = ref_hist[np.nonzero(ref_hist)]
    ref_ent = -np.sum(ref_hist * np.log(ref_hist))

    return ref_ent


# %%

class Clutter():
    def __init__(self, inputImage, numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=None, color_pool_sigma=3):

        self.inputImage = inputImage
        self.numlevels = numlevels
        self.contrast_filt_sigma = contrast_filt_sigma
        self.contrast_pool_sigma = 3 * contrast_filt_sigma if contrast_pool_sigma is None else contrast_pool_sigma
        self.color_pool_sigma = color_pool_sigma
        self.orient_pool_sigma = 7 / 2

        if isinstance(inputImage, str):
            self.im = cv2.imread(inputImage)
            if self.im is None:
                raise ValueError(f'Unable to open {inputImage} image file.')
            self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)

        elif isinstance(inputImage, np.ndarray):
            self.im = inputImage

        self.m, self.n, self.d = self.im.shape
        if self.d == 3:

            self.Lab = RGB2Lab(self.im)
            Lab_float = self.Lab.astype(np.float32)

            self.L, self.a, self.b = cv2.split(Lab_float)

            pyr = pt.pyramids.GaussianPyramid(self.L, height=self.numlevels)
            self.L_pyr = pyr.pyr_coeffs

            pyr = pt.pyramids.GaussianPyramid(self.a, height=self.numlevels)
            self.a_pyr = pyr.pyr_coeffs

            pyr = pt.pyramids.GaussianPyramid(self.b, height=self.numlevels)
            self.b_pyr = pyr.pyr_coeffs

            self.RRLab = [self.L_pyr, self.a_pyr, self.b_pyr]

        else:
            self.L = self.im
            pyr = pt.pyramids.GaussianPyramid(L, height=numlevels)
            self.L_pyr = pyr.pyr_coeffs
            print('Input image appears to be grayscale, so you can only use contrast clutter method\n')

    def collapse(self, clutter_levels):

        kernel_1d = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel_2d = conv2(kernel_1d, kernel_1d.T)

        clutter_map = clutter_levels[0]
        for scale in range(1, len(clutter_levels)):
            clutter_here = clutter_levels[scale]

            for kk in range(scale, 0, -1):
                clutter_here = pt.upConv(image=clutter_here, filt=kernel_2d, edge_type='reflect1', step=[2, 2],
                                         start=[0, 0])

            common_sz = min(clutter_map.shape[0], clutter_here.shape[0]), min(clutter_map.shape[1],
                                                                              clutter_here.shape[1])
            for i in range(0, common_sz[0]):
                for j in range(0, common_sz[1]):
                    clutter_map[i][j] = max(clutter_map[i][j], clutter_here[i][j])

        return clutter_map

    def display(self, method=''):

        if method == 'color':
            clutter_map = self.color_clutter_map
            clutter_levels = self.color_clutter_levels

        elif method == 'contrast':
            clutter_map = self.contrast_clutter_map
            clutter_levels = self.contrast_clutter_levels

        elif method == 'orientation':
            clutter_map = self.orientation_clutter_map
            clutter_levels = self.orientation_clutter_levels

        elif method == 'combine':
            clutter_levels = None
            clutter_map = self.clutter_map_fc

        else:
            raise ValueError(
                "method is not given or incorrect, should be selected from this list: ['color','contrast','orientation', 'combine']")

        min_min = np.mean(clutter_map)
        max_max = np.max(clutter_map)

        if clutter_levels is not None:
            numlevels = len(clutter_levels)
            size = clutter_map.shape[::-1]
            if numlevels > 8:
                raise ValueError('too many levels!!')

            for scale in range(numlevels):
                arr = clutter_levels[scale]
                new_arr = normlize(arr)
                new_PIL = Image.fromarray(new_arr)
                new_PIL = new_PIL.resize(size, Image.ANTIALIAS)
                new_PIL.save(f'{method} at level {scale}.png')

        new_arr = normlize(clutter_map)
        new_PIL = Image.fromarray(new_arr)
        new_PIL.save(f'collapsed {method} map.png')

    def computeColorClutter(self):

        covMx = {}
        self.color_clutter_levels = [0] * self.numlevels
        DL = [0] * self.numlevels
        Da = [0] * self.numlevels
        Db = [0] * self.numlevels

        deltaL2 = 0.0007 ** 2
        deltaa2 = 0.1 ** 2
        deltab2 = 0.05 ** 2

        bigG = RRgaussfilter1D(round(2 * self.color_pool_sigma), self.color_pool_sigma)

        for i in range(0, self.numlevels):
            DL[i] = RRoverlapconv(bigG, self.L_pyr[(i, 0)])
            DL[i] = RRoverlapconv(bigG.T, DL[i])  # E(L)
            Da[i] = RRoverlapconv(bigG, self.a_pyr[(i, 0)])
            Da[i] = RRoverlapconv(bigG.T, Da[i])  # E(a)
            Db[i] = RRoverlapconv(bigG, self.b_pyr[(i, 0)]);
            Db[i] = RRoverlapconv(bigG.T, Db[i])  # E(b)

            # dict idea
            covMx[(i, 0, 0)] = RRoverlapconv(bigG, self.L_pyr[(i, 0)] ** 2)
            covMx[(i, 0, 0)] = RRoverlapconv(bigG.T, covMx[(i, 0, 0)]) - DL[i] ** 2 + deltaL2  # cov(L,L) + deltaL2
            covMx[(i, 0, 1)] = RRoverlapconv(bigG, self.L_pyr[(i, 0)] * self.a_pyr[(i, 0)])
            covMx[(i, 0, 1)] = RRoverlapconv(bigG.T, covMx[(i, 0, 1)]) - DL[i] * Da[i]  # cov(L,a)
            covMx[(i, 0, 2)] = RRoverlapconv(bigG, self.L_pyr[(i, 0)] * self.b_pyr[(i, 0)])
            covMx[(i, 0, 2)] = RRoverlapconv(bigG.T, covMx[(i, 0, 2)]) - DL[i] * Db[i]  # cov(L,b)
            covMx[(i, 1, 1)] = RRoverlapconv(bigG, self.a_pyr[(i, 0)] ** 2)
            covMx[(i, 1, 1)] = RRoverlapconv(bigG.T, covMx[(i, 1, 1)]) - Da[i] ** 2 + deltaa2  # cov(a,a) + deltaa2
            covMx[(i, 1, 2)] = RRoverlapconv(bigG, self.a_pyr[(i, 0)] * self.b_pyr[(i, 0)])
            covMx[(i, 1, 2)] = RRoverlapconv(bigG.T, covMx[(i, 1, 2)]) - Da[i] * Db[i]  # cov(a,b)
            covMx[(i, 2, 2)] = RRoverlapconv(bigG, self.b_pyr[(i, 0)] ** 2)
            covMx[(i, 2, 2)] = RRoverlapconv(bigG.T, covMx[(i, 2, 2)]) - Db[i] ** 2 + deltab2;  # cov(b,b) + deltab2

            detIm = covMx[(i, 0, 0)] * (covMx[(i, 1, 1)] * covMx[(i, 2, 2)] - covMx[(i, 1, 2)] * covMx[(i, 1, 2)]) \
                    - covMx[(i, 0, 1)] * (covMx[(i, 0, 1)] * covMx[(i, 2, 2)] - covMx[(i, 1, 2)] * covMx[(i, 0, 2)]) \
                    + covMx[(i, 0, 2)] * (covMx[(i, 0, 1)] * covMx[(i, 1, 2)] - covMx[(i, 1, 1)] * covMx[(i, 0, 2)])

            self.color_clutter_levels[i] = np.sqrt(detIm) ** (1 / 3)
        return self.color_clutter_levels

    def colorClutter(self, color_pix=0):
        self.color_pix = color_pix
        self.color_clutter_levels = self.computeColorClutter()
        self.color_clutter_map = self.collapse(self.color_clutter_levels)

        if self.color_pix == 1:
            self.display(method='color')

        return self.color_clutter_levels, self.color_clutter_map

    def contrastClutter(self, contrast_pix=0):
        self.contrast_pix = contrast_pix
        contrast = RRcontrast1channel(self.L_pyr, self.contrast_filt_sigma)

        m, n = len(contrast), 1
        self.contrast_clutter_levels = [0] * m
        bigG = RRgaussfilter1D(round(self.color_pool_sigma * 2), self.color_pool_sigma)

        for scale in range(0, m):
            for channel in range(0, n):
                #         var(X) = E(X.^2) - E(X).^2
                #         get E(X) by filtering X with a 1-D Gaussian window separably in x and y directions
                meanD = RRoverlapconv(bigG, contrast[scale])
                meanD = RRoverlapconv(bigG.T, meanD)
                #         get E(X.^2) by filtering X.^2 with a 1-D Gaussian window separably in x and y directions
                meanD2 = RRoverlapconv(bigG, contrast[scale] ** 2)
                meanD2 = RRoverlapconv(bigG.T, meanD2)

                #         get variance by var(X) = E(X.^2) - E(X).^2
                stddevD = np.sqrt(abs(meanD2 - meanD ** 2))
                self.contrast_clutter_levels[scale] = stddevD

        self.contrast_clutter_map = self.collapse(self.contrast_clutter_levels)

        if self.contrast_pix == 1:
            self.display(method='contrast')

        return self.contrast_clutter_levels, self.contrast_clutter_map

    def RROrientationOppEnergy(self):
        """
        OPP_ENERGY    This runs the oriented opponent energy calculation that
        serves as the first stages in Bergen & Landy's (1990)
        texture segmentor, except it uses DOOG filters (which actually
        don't work as well, but at least we can more easily control the
        scale).
        """

        hvdd = [0] * self.numlevels
        hv = [0] * self.numlevels
        dd = [0] * self.numlevels
        out = [0] * self.numlevels
        total = [0] * self.numlevels

        noise = 1.0  # Was 1.5
        filterScale = 16 / 14 * 1.75
        poolScale = 1.75
        #     These probably seem like arbitrary numbers, but it's just trying to get
        #     three very different feature extraction methods to operate at basically
        #     the same scales.

        for scale in range(0, self.numlevels):
            #         Check this is the right order for Landy/Bergen. RRR
            hvdd[scale] = orient_filtnew(self.L_pyr[(scale, 0)], filterScale)
            #         filt with 4 oriented filters 0, 45, 90, 135.  Was sigma = 16/14, orient_filtnew,
            #         then 16/14*1.75 to match contrast and other scales.
            #         Eventually make this sigma a variable that's passed to this routine.
            #         hvdd[scale] is the 4 output images concatenated together,
            #         in the order horizontal, vertical, up-left, and down-right.

            hvdd[scale] = [x ** 2 for x in hvdd[scale]]  # local energy
            hvdd[scale] = poolnew(hvdd[scale],
                                  poolScale)  # Pools with a gaussian filter.  Was effectively sigma=1, then 1.75 to match 1.75 above.
            #         RRR Should look at these results and see if this is the right amount of
            #         pooling for the new filters.  It was right for the Landy-Bergen
            #         filters.
            hv[scale] = HV(hvdd[scale])  # get the difference image between horizontal and vertical: H-V (0-90)
            dd[scale] = DD(hvdd[scale])  # get the difference image between right and left: R-L (45-135)
            #         Normalize by the total response at this scale, assuming the total
            #         response is high enough.  If it's too low, we'll never see this
            #         orientation.  I'm not sure what to do here -- set it to zeros and
            #         it's like that's the orientation.  Maybe output the total response
            #         and decide what to do later.  RRR
            total[scale] = sumorients(hvdd[scale]) + noise  # add noise based upon sumorients at visibility threshold
            hv[scale] = hv[scale] / total[scale]  # normalize the hv and dd image
            dd[scale] = dd[scale] / total[scale]
            out[scale] = hv[scale], dd[
                scale]  # out is the 2 output images concatenated together, in the order of hv, dd

        return out

    def computeOrientationClutter(self):
        """
        computes the orientation clutter maps. Returns:
        clutter_levels, a cell structure, containing the orientation clutter at a 
        number of scales specified by numlevels;  -- cell(numlevels,1) --, the n'th 
        level of which can be accessed using clutter_levels{n}{1}

        input:
            L_pyr
            the Gaussian pyramid of L (from CIELab color space)
            the Gaussian pyramid is computed by alternately blurring and subsampling the L channels

        Orientation clutter is computed as the "volume" of an orientation distribution
        ellipsoid, which is the determinant of covariance matrix. Treats cos(2 theta)
        and sin(2 theta) (computed from OrientedOppEnergy) as a two-vector, and gets
        The covariance of this two-vector.  The covariance 
        matrix can be computed efficiently through linear filtering. More 
        specifically, cov(X,Y) = E(XY)-E(X)E(Y), where E (expectation value) 
        can be approximated by filtering with a Gaussian window. 
        poolScale is set to 7/2.

        Reference (though there is no orientation clutter in this reference):
        Ruth Rosenholtz, Yuanzhen Li, Jonathan Mansfield, and Zhenlan Jin. 
        Feature Congestion: A Measure of Display Clutter. CHI '05: Proc. of the SIGCHI conference 
        on Human factors in computing systems. May 2005. 761-770.  

        Based upon RRcomputeOrientationSaliency
        Ruth Rosenholtz, May 2006

        This currently seems far too dependent on luminance contrast.  Check into
        why this is so -- I thought we were normalizing by local contrast.

        """

        noise = 0.001  # Was eps, but that gave too much orientation noise in the saliency maps.  Then changed to 0.000001
        poolScale = 7 / 2

        numlevels = len(self.L_pyr);
        Dc = [0] * numlevels  # mean "cos 2 theta" at distractor scale
        Ds = [0] * numlevels  # mean "sin 2 theta" at distractor scale

        #     Get approximations to cos(2theta) and sin(2theta) from oriented opponent
        #     energy, at each of the numlevels of the pyramid
        angles = self.RROrientationOppEnergy()

        #     Compute the two-vector [meancos, meansin] at each scale, as well as the
        #     things we need to compute the mean and covariance of this two-vector at
        #     the larger, distractor scale.

        bigG = RRgaussfilter1D(round(8 * poolScale), 4 * poolScale)
        maxbigG = max(bigG) ** 2

        covMx = {}
        self.orientation_clutter_levels = [0] * numlevels

        for i in range(0, numlevels):
            cmx = angles[i][0]
            smx = angles[i][1]

            #         Pool to get means at distractor scale. In pooling, don't pool over the target
            #         region (implement this by pooling with a big Gaussian, then
            #         subtracting the pooling over the target region computed above.  Note,
            #         however, that we first need to scale the target region pooling so
            #         that its peak is the same height as this much broader Gaussian used
            #         to pool over the distractor region.

            Dc[i] = RRoverlapconv(bigG, cmx)
            Dc[i] = RRoverlapconv(bigG.T, Dc[i])
            Ds[i] = RRoverlapconv(bigG, smx)
            Ds[i] = RRoverlapconv(bigG.T, Ds[i])

            #         Covariance matrix elements.  Compare with computations in
            #         RRStatisticalSaliency.  I tried to match computeColorClutter, but I
            #         don't remember the meaning of some of the terms I removed.  XXX
            covMx[(i, 0, 0)] = RRoverlapconv(bigG, cmx ** 2)
            covMx[(i, 0, 0)] = RRoverlapconv(bigG.T, covMx[(i, 0, 0)]) - Dc[i] ** 2 + noise
            covMx[(i, 0, 1)] = RRoverlapconv(bigG, cmx * smx)
            covMx[(i, 0, 1)] = RRoverlapconv(bigG.T, covMx[(i, 0, 1)]) - Dc[i] * Ds[i]
            covMx[(i, 1, 1)] = RRoverlapconv(bigG, smx ** 2)
            covMx[(i, 1, 1)] = RRoverlapconv(bigG.T, covMx[(i, 1, 1)]) - Ds[i] ** 2 + noise

            #         Get determinant of covariance matrix, which is the volume of the
            #         covariance ellipse
            detIm = covMx[(i, 0, 0)] * covMx[(i, 1, 1)] - covMx[(i, 0, 1)] ** 2
            #         Take the square root considering variance is squared, and the square
            #         root again, since this is the area and the contrast measure is a "length"
            self.orientation_clutter_levels[i] = detIm ** (1 / 4)

        return self.orientation_clutter_levels

    def orientationClutter(self, orient_pix=0):
        self.orient_pix = orient_pix
        pool_sigma = 7 / 2

        self.orientation_clutter_levels = self.computeOrientationClutter()
        self.orientation_clutter_map = self.collapse(self.orientation_clutter_levels)

        if self.orient_pix == 1:
            self.display(method='orientation')

        return self.orientation_clutter_levels, self.orientation_clutter_map

    def computeClutter(self, color_pix=0, contrast_pix=0, orient_pix=0):
        # compute the color clutter
        color_clutter_levels, color_clutter_map = self.colorClutter(color_pix=color_pix)
        # compute the contrast clutter
        contrast_clutter_levels, contrast_clutter_map = self.contrastClutter(contrast_pix=contrast_pix)
        # compute the orientation clutter
        orient_clutter_levels, orientation_clutter_map = self.orientationClutter(orient_pix=orient_pix)

        # output them in cell structures
        color_clutter = [color_clutter_levels, color_clutter_map]
        contrast_clutter = [contrast_clutter_levels, contrast_clutter_map]
        orientation_clutter = [orient_clutter_levels, orientation_clutter_map]

        return color_clutter, contrast_clutter, orientation_clutter

    def getClutter_FC(self, p=1, pix=0):
        color_clutter, contrast_clutter, orient_clutter = self.computeClutter()
        self.clutter_map_fc = color_clutter[1] / 0.2088 + contrast_clutter[1] / 0.0660 + orient_clutter[1] / 0.0269
        self.clutter_scalar_fc = np.mean(self.clutter_map_fc ** p) ** (1 / p)  # element wise

        if pix == 1:
            self.display(method='combine')
        return self.clutter_scalar_fc, self.clutter_map_fc

    def band_entropy(self, map_, wlevels, wor):
        """
        en_band = band_entropy(map_, wlevels, wor)
        Inputs: 
            "map_": a monochromatic image
            "wlevels": the number of spatial scales for the subband decomposition
            "wor": the number of orientations for the subband decomposition
        Outputs:
            "en_band": a vector containing Shannon entropies of all the subbands

        Reference: Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano. "Measuring
        Visual Clutter". To appear in Journal of Vision, 7(2).

        Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano, March 2007.
        """

        # decompose the image into subbands:
        self.SFpyr = pt.pyramids.SteerablePyramidFreq(map_, height=wlevels, order=wor - 1)
        S = self.SFpyr.pyr_coeffs

        en_band = []
        for ind in S.keys():
            en_band.append(entropy(S[ind].ravel()))

        return en_band

    def getClutter_SE(self, wlevels=3, wght_chrom=0.0625):
        """
        [clutter_se] = getClutter_SE(map_name, [wlevels], [wght_chrom])
        Subband Entropy measure of visual clutter.
        Outputs:
            "clutter_se": the subband entropy clutter of the image.
        Inputs: 
            "map_name": input image. It can be a string (file name of the image),
            or an array (the image itself).
            "wlevels":  the number of scales (optional, default 3)
            "wght_chrom": the weight on chrominance (optional, default 0.0625)

        This measure (Subband Entropy) of visual clutter is based on the notion
        that clutter is related to the number of bits required for subband
        (wavelet) image coding.

        Reference: 
        Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano. "Measuring Visual Clutter". 
        Journal of Vision, 7(2), 2007. http://www.journalofvision.com/7/2/

        Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano, March 2007.
        """

        wor = 4;
        #      luminance channel
        en_band = self.band_entropy(self.L, wlevels, wor)
        clutter_se = np.mean(en_band)

        if self.d == 1:
            return clutter_se

        #     chrominance channels:
        for jj in [self.a, self.b]:
            if np.max(jj) - np.min(jj) < 0.008:
                jj = np.zeros_like(jj)

            en_band = self.band_entropy(jj, wlevels, wor)
            clutter_se = clutter_se + wght_chrom * np.mean(en_band)

        clutter_se = clutter_se / (1 + 2 * wght_chrom)
        return clutter_se


# %%

clt = Clutter('test.jpg', numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=3, color_pool_sigma=3)

# %%

color_clutter = clt.colorClutter(color_pix=1)

# %%

contrast_clutter = clt.contrastClutter(contrast_pix=1)

# %%

orientation_clutter = clt.orientationClutter(orient_pix=1)

# %%

color_clutter, contrast_clutter, orientation_clutter = clt.computeClutter(color_pix=1, contrast_pix=1, orient_pix=1)

# %%

clutter_scalar_fc, clutter_map_fc = clt.getClutter_FC(p=1, pix=1)

# %%

clutter_scalar_fc

# %%

clt.getClutter_SE(wlevels=3, wght_chrom=0.0625)
