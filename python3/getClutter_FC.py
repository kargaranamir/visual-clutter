import cv2
# import pyrtools as pt
import numpy as np



def computeClutter(inputImage, numlevels=3, contrast_filt_sigma=1, contrast_pool_sigma=None, color_pool_sigma=3, contrast_pix=0, color_pix=0, orient_pix=0):
    if contrast_pool_sigma is None:
        contrast_pool_sigma = 3 * contrast_filt_sigma

    orient_pool_sigma = 7/2

    if isinstance(inputImage, str):
        im = cv2.imread(inputImage)
        if im is None:
            print('Unable to open %s image file.') #TODO: add logger
        else:
            m, n, d = im.shape
            if d == 3:
                Lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
                RRLab = np.zeros_like(Lab)

                # pyr = pt.pyramids.GaussianPyramid(Lab[:,:,0], height=numlevels) # approved
                # RRLab[:,:,0] = pyr.recon_pyr()
                #
                # pyr = pt.pyramids.GaussianPyramid(Lab[:,:,1], height=numlevels)
                # RRLab[:,:,1] = pyr.recon_pyr()
                #
                # pyr = pt.pyramids.GaussianPyramid(Lab[:,:,2], height=numlevels)
                # RRLab[:,:,2] = pyr.recon_pyr()

    # return color_clutter, contrast_clutter, orientation_clutter

# def getClutter_Fc(filename, p=1):
#     color_clutter, contrast_clutter, orient_clutter = computeClutter(filename, 3, 1, 3, 3, 0, 0, 0)
#     clutter_map_fc = color_clutter{2} / 0.2088 + contrast_clutter{2} / 0.0660 + orient_clutter{2} / 0.0269
#     clutter_scalar_fc = mean(clutter_map_fc(:).^ p).^ (1 / p) #element wise
#     return clutter_scalar_fc, clutter_map_fc


computeClutter('test.jpg')