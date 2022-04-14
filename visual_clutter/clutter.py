#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pyrtools as pt
from PIL import Image
from visual_clutter.utils import *


class Vlc():
    """
    
    Class of two measures of visual clutter (Feature Congestion and Subband Entropy)
    
    Parameters
    ---------- 
    inputImage : gives the input. It can be one of the following 2 things: 1. an RGB image; 2. a string, i.e., file name of an RGB image.
    numlevels : the number of levels.
    contrast_filt_sigma : the sigma (standard deviation) of the center-surround DoG1 filter used for computing the contrast 
    contrast_pool_sigma : the sigma (standard deviation) of this Gaussian window for contrast clutter. Default = 3*filt_sigma. 
    color_pool_sigma : the sigma (standard deviation) of this Gaussian window for color clutter, Defaults to 3.
    
    Methods
    -------
    getClutter_FC: computes Feature Congestion clutter, outputs both a scalar (clutter of the whole image) and a map (local clutter).
    getClutter_SE: computes Subband Entropy clutter, outputs only a scalar.
    colorClutter: computes clutter maps indicating local variability in color
    contrastClutter: computes clutter maps indicating local variability in contrast
    orientationClutter: computes clutter maps indicating local variability in orientation
 
    (Please see individual routines for more info about parameters and outputs.)

    References
    ----------
    Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano. "Measuring Visual Clutter". 
    Journal of Vision, 7(2), 2007. http://www.journalofvision.com/7/2/
    Ruth Rosenholtz, Yuanzhen Li, and Lisa Nakano, March 2007.
    
    Ruth Rosenholtz, Yuanzhen Li, Jonathan Mansfield, and Zhenlan Jin. "Feature Congestion: A Measure of Display Clutter".
    CHI '05: Proc. of the SIGCHI conference on Human factors in computing systems. May 2005. 761-770. 
    
    """
    
    def __init__(self, inputImage, numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=None, color_pool_sigma=3, output_dir='.', prefix='test'):
        self.inputImage = inputImage
        self.numlevels = numlevels
        self.contrast_filt_sigma = contrast_filt_sigma
        self.contrast_pool_sigma = 3 * contrast_filt_sigma if contrast_pool_sigma is None else contrast_pool_sigma
        self.color_pool_sigma = color_pool_sigma
        # orient_pool_sigma is the sigma (standard deviation) of this Gaussian window, and here is hard-wired to 7/2.
        self.orient_pool_sigma = 7/2
        
        # address of output images
        self.dir = output_dir + '/' if output_dir[-1]!='/' else output_dir
        self.prefix = prefix
        
        if isinstance(inputImage, str): 
            self.im = cv2.imread(inputImage)
            if self.im is None:
                raise ValueError(f'Unable to open {inputImage} image file.')     
            self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            
        elif isinstance(inputImage,np.ndarray):
            self.im = inputImage
        

        # The input image, im, had better be a MxNx3 matrix, in which case we assume it is an RGB image.
        # If it's MxN, it's probably gray, and color clutter method is not appropriate.        
        number_of_dimension = len(self.im.shape)
        if number_of_dimension == 3: 
            self.m, self.n, self.d = self.im.shape
        elif number_of_dimension == 2: 
            self.m, self.n = self.im.shape
            self.d = 1
        else: 
            raise ValueError(f'inputImage should be as one of these formats: MxNxD or MxN')    
             
        if self.d == 3:
            # we first convert it into the perceptually-based CIELab color space.
            self.Lab = RGB2Lab(self.im)
            Lab_float = self.Lab.astype(np.float32)
            # luminance(L) and the chrominance(a,b) channels
            self.L, self.a, self.b = cv2.split(Lab_float)

            # Get Gaussian pyramids (one for each of L,a,b)
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
            print ('Input image appears to be grayscale, so you can only use contrast clutter method\n')
            
    
    
    def collapse(self, clutter_levels):
        """
        
        Collapse over scales by taking the maximum.
        
        Notes
        -----
        first get a Gaussian kernel to upsample the clutter maps on bigger scales
        so that the clutter maps would have the same sizes, and max can be taken
        across scales.
        
        """
        kernel_1d = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel_2d = conv2(kernel_1d, kernel_1d.T)

        clutter_map = clutter_levels[0].copy()
        for scale in range(1,len(clutter_levels)):
            clutter_here = clutter_levels[scale]

            for kk in range(scale, 0, -1):
                clutter_here = pt.upConv(image=clutter_here, filt=kernel_2d, edge_type='reflect1', step=[2,2], start=[0,0])

            common_sz = min(clutter_map.shape[0], clutter_here.shape[0]), min(clutter_map.shape[1], clutter_here.shape[1])
            for i in range(0, common_sz[0]):
                for j in range(0, common_sz[1]):
                    clutter_map[i][j] = max(clutter_map[i][j], clutter_here[i][j])

        return clutter_map




    def display(self, method=''):
        """
        
        Saves the clutter map(s)
        
        """
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
            raise ValueError("method is not given or incorrect, should be selected from this list: ['color','contrast','orientation', 'combine']")
            
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
                
                # save clutter level(s) 
                new_PIL.save(f'{self.dir}{self.prefix}_{method}_at_level_{scale+1}.png')

        new_arr = normlize(clutter_map)
        new_PIL = Image.fromarray(new_arr)
        
        # save collapsed clutter map(s)
        new_PIL.save(f'{self.dir}{self.prefix}_collapsed_{method}_map.png')


    def computeColorClutter(self):
        """
        
        Computes the color clutter maps. 
            
        Returns
        -------
        color_clutter_levels : a list structure, containing the color clutter at a 
        number of scales specified by numlevels.
        the n'th level of which can be accessed using color_clutter_levels[n], n starts from 0 to numlevels-1

        Notes
        -----
        Color clutter is computed as the "volume" of a color distribution
        ellipsoid, which is the determinant of covariance matrix. Covariance 
        matrix can be computed efficiently through linear filtering. More 
        specifically, cov(X,Y) = E(XY)-E(X)E(Y), where E (expectation value) 
        can be approximated by filtering with a Gaussian window. 
        """
        
        # initiatialization
        covMx = {}
        self.color_clutter_levels = [0] * self.numlevels
        DL = [0] * self.numlevels
        Da = [0] * self.numlevels
        Db = [0] * self.numlevels


        # sensitivitis to the L,a,and b channels are different, therefore we use
        # deltaL2, deltaa2, and deltab2 to "scale" the L,a,b axes when computing
        # the covariance matrix. Eventually these numbers should be vary according
        # to the spatial scales, mimicing our visual system's sensitivity function
        deltaL2 = 0.0007 ** 2
        deltaa2 = 0.1 ** 2
        deltab2 = 0.05 ** 2

        # Get a Gaussian filter for computing the covariance
        bigG = RRgaussfilter1D(round(2*self.color_pool_sigma), self.color_pool_sigma)

        for i in range(0, self.numlevels):
            # get E(X) by filtering X with a 1-D Gaussian window separably in x and y directions:
            DL[i] = RRoverlapconv(bigG, self.L_pyr[(i,0)])
            DL[i] = RRoverlapconv(bigG.T, DL[i])   # E(L)
            Da[i] = RRoverlapconv(bigG, self.a_pyr[(i,0)])
            Da[i] = RRoverlapconv(bigG.T, Da[i])   # E(a)
            Db[i] = RRoverlapconv(bigG, self.b_pyr[(i,0)]);
            Db[i] = RRoverlapconv(bigG.T, Db[i])    # E(b)


            # Covariance matrix 
            # covMx(L,a,b) = | cov(L,L)  cov(L,a)  cov(L,b) |
            #                | cov(a,L)  cov(a,a)  cov(a,b) |
            #                | cov(b,L)  cov(b,a)  cov(b,b) |
            # where cov(X,Y) = E(XY) - E(X)E(Y)
            #   and if X is the same as Y, then it's the variance var(X) =
            #   E(X.^2)-E(X).^2
            # and as cov(X,Y) = cov(Y,X), covMx is symmetric
            # covariance matrix elements:
            covMx[(i,0,0)] = RRoverlapconv(bigG, self.L_pyr[(i,0)] ** 2)
            covMx[(i,0,0)] = RRoverlapconv(bigG.T, covMx[(i,0,0)]) - DL[i] ** 2 + deltaL2  # cov(L,L) + deltaL2
            covMx[(i,0,1)] = RRoverlapconv(bigG, self.L_pyr[(i,0)] * self.a_pyr[(i,0)])
            covMx[(i,0,1)] = RRoverlapconv(bigG.T, covMx[(i,0,1)]) - DL[i] * Da[i]        # cov(L,a)
            covMx[(i,0,2)] = RRoverlapconv(bigG, self.L_pyr[(i,0)] * self.b_pyr[(i,0)])
            covMx[(i,0,2)] = RRoverlapconv(bigG.T, covMx[(i,0,2)]) - DL[i] * Db[i]        # cov(L,b)
            covMx[(i,1,1)] = RRoverlapconv(bigG, self.a_pyr[(i,0)] ** 2)
            covMx[(i,1,1)] = RRoverlapconv(bigG.T, covMx[(i,1,1)]) - Da[i] ** 2 + deltaa2  # cov(a,a) + deltaa2
            covMx[(i,1,2)] = RRoverlapconv(bigG, self.a_pyr[(i,0)] * self.b_pyr[(i,0)])
            covMx[(i,1,2)] = RRoverlapconv(bigG.T, covMx[(i,1,2)]) - Da[i] * Db[i]        # cov(a,b)
            covMx[(i,2,2)] = RRoverlapconv(bigG, self.b_pyr[(i,0)] ** 2)    
            covMx[(i,2,2)] = RRoverlapconv(bigG.T, covMx[(i,2,2)]) - Db[i] ** 2 + deltab2;  # cov(b,b) + deltab2

            # Get the determinant of covariance matrix
            # which is the "volume" of the covariance ellipsoid
            detIm = covMx[(i,0,0)]*(covMx[(i,1,1)]*covMx[(i,2,2)]-covMx[(i,1,2)]*covMx[(i,1,2)])            - covMx[(i,0,1)]*(covMx[(i,0,1)]*covMx[(i,2,2)]-covMx[(i,1,2)]*covMx[(i,0,2)])            + covMx[(i,0,2)]*(covMx[(i,0,1)]*covMx[(i,1,2)]-covMx[(i,1,1)]*covMx[(i,0,2)])

            # take the square root considering variance is squared, and the cube
            # root, since this is the volume and the contrast measure is a "length"
            self.color_clutter_levels[i] = np.sqrt(detIm) ** (1/3)
        return self.color_clutter_levels
    

    def colorClutter(self, color_pix=0):
        """
        
        Computes the color clutter map(s) of an image. 
         
        Parameters
        ---------- 
        color_pix : if it is 1 then saves the color clutter map(s) as png files  If it's 0, does not save
         (useful for batch processing of many images).  Defaults not to save.

        Returns
        -------
        color_clutter_levels : a list structure, containing the color clutter at a number 
        of scales specified by numlevels; the n'th level of which can be accessed using color_clutter_levels[n], n starts from 0 to numlevels-1
        color_clutter_map : an array of the same size as inputImage, is a single clutter map 
        collapsed from color_clutter_levels, which is the clutter measure at multiple scales
        now the "collapsing" is done by taking the maximal values across scales
 
        Notes
        -----
        Color clutter is computed as the "volume" of a color distribution
        ellipsoid, which is the determinant of covariance matrix. Covariance 
        matrix can be computed efficiently through linear filtering. More 
        specifically, cov(X,Y) = E(XY)-E(X)E(Y), where E (expectation value) 
        can be approximated by filtering with a Gaussian window. 
        
        """
        self.color_pix = color_pix
        
        # Compute clutter
        self.color_clutter_levels = self.computeColorClutter()
        self.color_clutter_map = self.collapse(self.color_clutter_levels)
        
        # to save the clutter maps:       
        if self.color_pix==1:
            self.display(method='color')

        return self.color_clutter_levels, self.color_clutter_map

    
    def contrastClutter(self, contrast_pix=0):  
        """
        
        Computes the contrast clutter map(s) of an image.
        
        Parameters
        ---------- 
        contrast_pix : if it is 1 then saves the contrast clutter map(s) as png files  If it's 0, does not save
         (useful for batch processing of many images).  Defaults to not save.

        Returns
        -------
        contrast_clutter_levels : a list structure, containing the orientation clutter at a number 
        of scales specified by numlevels; the n'th level of which can be accessed using contrast_clutter_levels[n], n starts from 0 to numlevels-1
        contrast_clutter_map : an array of the same size as inputImage, is a single clutter map 
        collapsed from contrast_clutter_levels, which is the clutter measure at multiple scales
        now the "collapsing" is done by taking the maximal values across scales
 
        """
        self.contrast_pix = contrast_pix
        # We then compute a form of "contrast-energy" by filtering the luminance
        # channel L by a center-surround filter and squaring (or taking the absolute 
        # values of) the filter outputs. The center-surround filter is a DoG1 filter 
        # with std 'contrast_filt_sigma'.
        contrast = RRcontrast1channel(self.L_pyr, self.contrast_filt_sigma)

        # initiate clutter_map and clutter_levels:
        m, n = len(contrast), 1
        self.contrast_clutter_levels = [0] * m
        
        # Get a Gaussian filter for computing the variance of contrast
        # Since we used a Gaussian pyramid to find contrast features, these filters 
        # have the same size regardless of the scale of processing.
        bigG = RRgaussfilter1D(round(self.contrast_pool_sigma*2), self.contrast_pool_sigma)

        for scale in range(0,m):
            for channel in range(0,n):
                # var(X) = E(X.^2) - E(X).^2
                # get E(X) by filtering X with a 1-D Gaussian window separably in x and y directions
                meanD = RRoverlapconv(bigG, contrast[scale])
                meanD = RRoverlapconv(bigG.T, meanD)
                # get E(X.^2) by filtering X.^2 with a 1-D Gaussian window separably in x and y directions
                meanD2 = RRoverlapconv(bigG, contrast[scale] ** 2)
                meanD2 = RRoverlapconv(bigG.T, meanD2)

                # get variance by var(X) = E(X.^2) - E(X).^2
                stddevD = np.sqrt(abs(meanD2 - meanD ** 2))
                self.contrast_clutter_levels[scale] = stddevD

        self.contrast_clutter_map = self.collapse(self.contrast_clutter_levels)
        
        
        if self.contrast_pix==1:
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

        noise = 1.0    # Was 1.5
        filterScale = 16/14*1.75
        poolScale = 1.75
        # These probably seem like arbitrary numbers, but it's just trying to get
        # three very different feature extraction methods to operate at basically
        # the same scales.


        for scale in range(0, self.numlevels):
            # Check this is the right order for Landy/Bergen. RRR
            hvdd[scale] = orient_filtnew(self.L_pyr[(scale,0)],filterScale) 
            # filt with 4 oriented filters 0, 45, 90, 135.  Was sigma = 16/14, orient_filtnew,
            # then 16/14*1.75 to match contrast and other scales.
            # Eventually make this sigma a variable that's passed to this routine.
            # hvdd[scale] is the 4 output images concatenated together, 
            # in the order horizontal, vertical, up-left, and down-right.

            hvdd[scale] = [x ** 2 for x in hvdd[scale]]    #local energy
            hvdd[scale] = poolnew(hvdd[scale], poolScale) #Pools with a gaussian filter.  Was effectively sigma=1, then 1.75 to match 1.75 above.
            # RRR Should look at these results and see if this is the right amount of
            # pooling for the new filters.  It was right for the Landy-Bergen
            # filters.
            hv[scale] = HV(hvdd[scale]) # get the difference image between horizontal and vertical: H-V (0-90)
            dd[scale] = DD(hvdd[scale]) # get the difference image between right and left: R-L (45-135)
            # Normalize by the total response at this scale, assuming the total
            # response is high enough.  If it's too low, we'll never see this
            # orientation.  I'm not sure what to do here -- set it to zeros and
            # it's like that's the orientation.  Maybe output the total response
            # and decide what to do later.  RRR
            total[scale] = sumorients(hvdd[scale]) + noise # add noise based upon sumorients at visibility threshold
            hv[scale] = hv[scale]/total[scale] # normalize the hv and dd image
            dd[scale] = dd[scale]/total[scale]
            out[scale] = hv[scale], dd[scale] # out is the 2 output images concatenated together, in the order of hv, dd

        return out


    def computeOrientationClutter(self):
        """
        
        Computes the orientation clutter maps. 
            
        Returns
        -------
        orientation_clutter_levels : a list structure, containing the orientation clutter at a 
        number of scales specified by numlevels.
        the n'th level of which can be accessed using orientation_clutter_levels[n], n starts from 0 to numlevels-1

        Notes
        -----
        Orientation clutter is computed as the "volume" of an orientation distribution
        ellipsoid, which is the determinant of covariance matrix. Treats cos(2 theta)
        and sin(2 theta) (computed from OrientedOppEnergy) as a two-vector, and gets
        The covariance of this two-vector.  The covariance 
        matrix can be computed efficiently through linear filtering. More 
        specifically, cov(X,Y) = E(XY)-E(X)E(Y), where E (expectation value) 
        can be approximated by filtering with a Gaussian window. 
        poolScale is set to 7/2.

        This currently seems far too dependent on luminance contrast.  Check into
        why this is so -- I thought we were normalizing by local contrast.

        """

        noise = 0.001  # Was eps, but that gave too much orientation noise in the saliency maps.  Then changed to 0.000001
        poolScale = 7/2

        numlevels = len(self.L_pyr);
        Dc = [0] * numlevels  # mean "cos 2 theta" at distractor scale
        Ds = [0] * numlevels  # mean "sin 2 theta" at distractor scale

        # Get approximations to cos(2theta) and sin(2theta) from oriented opponent
        # energy, at each of the numlevels of the pyramid
        angles = self.RROrientationOppEnergy()

        # Compute the two-vector [meancos, meansin] at each scale, as well as the
        # things we need to compute the mean and covariance of this two-vector at
        # the larger, distractor scale.
        bigG = RRgaussfilter1D(round(8*poolScale), 4*poolScale)
        maxbigG = max(bigG) ** 2


        covMx = {}
        self.orientation_clutter_levels = [0] * numlevels

        for i in range(0,numlevels):
            cmx = angles[i][0]
            smx = angles[i][1]

            # Pool to get means at distractor scale. In pooling, don't pool over the target
            # region (implement this by pooling with a big Gaussian, then
            # subtracting the pooling over the target region computed above.  Note,
            # however, that we first need to scale the target region pooling so
            # that its peak is the same height as this much broader Gaussian used
            # to pool over the distractor region.
            Dc[i] = RRoverlapconv(bigG, cmx)
            Dc[i] = RRoverlapconv(bigG.T, Dc[i])
            Ds[i] = RRoverlapconv(bigG, smx)
            Ds[i] = RRoverlapconv(bigG.T, Ds[i])

            # Covariance matrix elements.  Compare with computations in
            # RRStatisticalSaliency.  I tried to match computeColorClutter, but I
            # don't remember the meaning of some of the terms I removed.  XXX
            covMx[(i,0,0)] = RRoverlapconv(bigG, cmx ** 2)
            covMx[(i,0,0)] = RRoverlapconv(bigG.T, covMx[(i,0,0)]) - Dc[i] ** 2 + noise
            covMx[(i,0,1)] = RRoverlapconv(bigG, cmx * smx)
            covMx[(i,0,1)] = RRoverlapconv(bigG.T, covMx[(i,0,1)]) - Dc[i] * Ds[i]
            covMx[(i,1,1)] = RRoverlapconv(bigG, smx ** 2)
            covMx[(i,1,1)] = RRoverlapconv(bigG.T, covMx[(i,1,1)]) - Ds[i] ** 2 + noise

            # Get determinant of covariance matrix, which is the volume of the
            # covariance ellipse
            detIm = covMx[(i,0,0)] * covMx[(i,1,1)] - covMx[(i,0,1)] ** 2
            # Take the square root considering variance is squared, and the square
            # root again, since this is the area and the contrast measure is a "length"
            self.orientation_clutter_levels[i] = detIm ** (1/4)

        return self.orientation_clutter_levels
    

    def orientationClutter(self, orient_pix=0):
        """
        Computes the orientation clutter map(s) of an image. 
         
        Parameters
        ---------- 
        orient_pix : if it is 1 then saves the orientation clutter map(s) as png files  If it's 0, does not save
         (useful for batch processing of many images).  Defaults not to save.

        Returns
        -------
        orientation_clutter_levels : a list structure, containing the orientation clutter at a number 
        of scales specified by numlevels; the n'th level of which can be accessed using orientation_clutter_levels[n], n starts from 0 to numlevels-1
        orientation_clutter_map : an array of the same size as inputImage, is a single clutter map 
        collapsed from orientation_clutter_levels, which is the clutter measure at multiple scales
        now the "collapsing" is done by taking the maximal values across scales
 
        Notes
        -----
        Orientation clutter is computed as the "volume" of an orientation distribution
        ellipsoid, which is the determinant of covariance matrix. Covariance 
        matrix can be computed efficiently through linear filtering. More 
        specifically, cov(X,Y) = E(XY)-E(X)E(Y), where E (expectation value) 
        can be approximated by filtering with a Gaussian window. 
        
        """
        
        self.orient_pix = orient_pix
        #  Compute clutter
        self.orientation_clutter_levels = self.computeOrientationClutter()
        self.orientation_clutter_map = self.collapse(self.orientation_clutter_levels)
        
        # to save the clutter maps:
        if self.orient_pix==1:
            self.display(method='orientation')

        return self.orientation_clutter_levels, self.orientation_clutter_map
    
    def computeClutter(self, color_pix=0, contrast_pix=0, orient_pix=0) -> tuple:
        """        
        
        Computes Feature Congestion clutter map(s) of an image.

        Parameters
        ---------- 
        color_pix :if it is 1 then saves the color clutter map(s) as png files
        contrast_pix : if it is 1 then saves the contrast clutter map(s) as png files
        orient_pix : if it is 1 then saves the orientation clutter map(s) as png files
        
        Returns
        -------
        color_clutter : a list structure containing the color clutter map(s)
        contrast_clutter: a list structure containing the contrast clutter map(s)
        orientation_clutter: a list structure containing the orientation clutter map(s)
        
        Notes
        -----
        As for each of the three list structure, *[0] is another list structure containing 
        the clutter maps at a number of scales specified by "numlevels", and *[1] is a 
        single clutter map (same size as the input image) collapsed from all scales
  
        Examples
        --------
        >>> clt = Vlc('test.jpg', numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=3, color_pool_sigma=3)
        >>> color_clutter, contrast_clutter, orientation_clutter = clt.computeClutter(color_pix=1, contrast_pix=1, orient_pix=1)
  
        """

        # compute the color clutter
        color_clutter_levels, color_clutter_map = self.colorClutter(color_pix = color_pix)
        # compute the contrast clutter
        contrast_clutter_levels, contrast_clutter_map = self.contrastClutter(contrast_pix = contrast_pix)
        # compute the orientation clutter
        orient_clutter_levels, orientation_clutter_map = self.orientationClutter(orient_pix = orient_pix)

        # output them in list structures
        color_clutter = [color_clutter_levels, color_clutter_map]
        contrast_clutter = [contrast_clutter_levels, contrast_clutter_map]
        orientation_clutter = [orient_clutter_levels, orientation_clutter_map]

        return color_clutter, contrast_clutter, orientation_clutter
    
    
    def getClutter_FC(self, p=1, pix=0) -> (float, np.array):
        """

        Computes Feature Congestion measure of visual clutter.
        
        Parameters
        ---------- 
        p : a parameter when combining local clutter over 
        space; the combination can be considered Minkowski distance of order p
        pix : if it is 1 then saves the outputs as png files

        Returns
        -------
        clutter_scalar_fc : is a scalar, which gives the Feature Congestion clutter of the whole image.
        clutter_map_fc : is a clutter map (same size as the input image), which gives local clutter information.

        Notes
        -----
        This measure (Feature Congestion) of visual clutter is related to the
        local variability in certain key features, e.g., color, contrast, and orientation.
    
        Examples
        --------
        >>> clt = Vlc('test.jpg', numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=3, color_pool_sigma=3)
        >>> clutter_scalar_fc, clutter_map_fc = clt.getClutter_FC(p=1, pix=1)
        
        """
        color_clutter, contrast_clutter, orient_clutter = self.computeClutter(pix, pix, pix)
        self.clutter_map_fc = color_clutter[1] / 0.2088 + contrast_clutter[1] / 0.0660 + orient_clutter[1] / 0.0269
        self.clutter_scalar_fc = np.mean(self.clutter_map_fc ** p) ** (1 / p) #element wise
        
        if pix==1:
            self.display(method='combine')
        return self.clutter_scalar_fc, self.clutter_map_fc


    def band_entropy(self, map_, wlevels, wor) -> list:
        """
        
        Parameters
        ---------- 
        map_ : a monochromatic image
        wlevels : the number of spatial scales for the subband decomposition
        wor : the number of orientations for the subband decomposition
        
        Returns
        -------
        en_band : a vector containing Shannon entropies of all the subbands
        
        Examples
        --------
        # luminance channel:
        >>> clt = Vlc('test.jpg')
        >>> l_channel = clt.L
        >>> en_band = clt.band_entropy(l_channel, wlevels=3, wor=4)
        
        """
        
        # decompose the image into subbands:
        self.SFpyr = pt.pyramids.SteerablePyramidFreq(map_, height=wlevels, order=wor-1)
        S = self.SFpyr.pyr_coeffs
        
        en_band = []
        for ind in S.keys():
            en_band.append(entropy(S[ind].ravel()))
            

        return en_band
    
    
    def getClutter_SE(self, wlevels=3, wght_chrom=0.0625) -> float:
        """
        
        Computes the subband entropy measure of visual clutter.

        Parameters
        ----------
        wlevels : the number of scales (optional, default 3)
        wght_chrom : the weight on chrominance (optional, default 0.0625)

        Returns
        -------
        clutter_se : float, the subband entropy clutter of the image.

        Notes
        -----
        This measure (Subband Entropy) of visual clutter is based on the notion
        that clutter is related to the number of bits required for subband
        (wavelet) image coding.
        
        Examples
        --------
        >>> clt = Vlc('test.jpg')
        >>> clutter_se = clt.getClutter_SE(wlevels=3, wght_chrom=0.0625)
        
        """

        wor = 4
        # luminance channel:
        en_band = self.band_entropy(self.L, wlevels, wor)
        clutter_se = np.mean(en_band)

        if self.d == 1:
            return clutter_se

        # chrominance channels:
        for jj in [self.a, self.b]:
            if np.max(jj)-np.min(jj) < 0.008:
                jj = np.zeros_like(jj)

            en_band = self.band_entropy(jj, wlevels, wor)
            clutter_se = clutter_se + wght_chrom * np.mean(en_band)

        clutter_se = clutter_se/(1 + 2 * wght_chrom)
        return clutter_se
