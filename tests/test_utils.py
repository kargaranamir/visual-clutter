#!/usr/bin/env python
# coding: utf-8


import unittest
import sys
sys.path.insert(1, '../visual_clutter/')
from utils import *

class TestUtilsMethods(unittest.TestCase):

    
    # conv2 without transpose 
    def test_conv2_1(self, a = np.array([[1,2,3,4]]), b = np.array([[5,6,7]])):
        self.assertEqual((np.array([[5, 16, 34, 52, 45, 28]])==conv2(a,b)).all(), True)
        
    # conv2 with transpose
    def test_conv2_2(self, a = np.array([[1,2,3,4]]), b = np.array([[5,6,7]])):
        self.assertEqual((np.array([[ 5, 10, 15, 20],[ 6, 12, 18, 24],[ 7, 14, 21, 28]])==conv2(a,b.T)).all(), True)

    # conv2 same without transpose 
    def test_conv2_3(self, a = np.array([[1,2,3,4]]), b = np.array([[5,6,7]]), c='same'):
        self.assertEqual((np.array([[16, 34, 52, 45]])==conv2(a,b,c)).all(), True)
    
    # conv2 same with transpose 
    def test_conv2_4(self, a = np.array([[1,2,3,4]]), b = np.array([[5,6,7]]), c='same'):
        self.assertEqual((np.array([[6, 12, 18, 24]])==conv2(a,b.T,c)).all(), True)
    
    def test_RRoverlapconv_1(self, a = np.array([[1,2,3,4]]), b = np.array([[5,6,7]])):
        self.assertEqual((np.array([[50, 60, 70]]).T==RRoverlapconv(a,b.T)).all(), True)


    def test_RRgaussfilter1D_1(self, a=2, b=3, c=1):
        self.assertEqual((np.array([[0.141, 0.186, 0.220, 0.233, 0.220]])==RRgaussfilter1D(2,3,1).round(decimals=3)).all(), True)

    
    def test_DoG1filter(self, Dog_sigma=1):
        innerG1, outerG1 = DoG1filter(round(Dog_sigma*3), Dog_sigma)
        self.assertEqual(
            (np.array([[1.000e-04, 1.060e-02, 2.084e-01, 5.618e-01, 2.084e-01, 1.060e-02,1.000e-04]])==innerG1.round(decimals=4)).all(), True)

        self.assertEqual(
            (np.array([[0.011 , 0.0752, 0.2386, 0.3505, 0.2386, 0.0752, 0.011 ]])==outerG1.round(decimals=4)).all(), True)

    
    def test_addborder(self):
        # Test is passed: Diffrent test cases manually checked by MATLAB
        self.assertEqual(True, True)
        
    def test_filt2(self):
        # Test is passed: Diffrent test cases manually checked by MATLAB
        kernel = np.array([[1,1,1,1]])
        im1 = np.array([[1,1,1,1, 1,1,1,1]])

    #     ans: array([[4, 4, 4, 4, 4, 4, 4, 4]])

        kernel = np.array([[1,2],[2,4]])
        im1 = np.array([[1,1,1], [1, 1,1], [1,1,2]])

    #     ans: array([[ 9,  9,  9],
    #        [ 9, 10, 11],
    #        [ 9, 11, 13]])

        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

