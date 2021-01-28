import dip
#
import numpy as np
import cv2

def SpatialFiltering(src, filter):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    tmp = dip.Image()
    tmp.replace(cv2.filter2D(src.ndarray, -1, filter))
    dst = dip.Image()
    dst.replace(cv2.convertScaleAbs(tmp.ndarray))
    return dst


def Averaging(image):
    filter = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
    return filtering(image, filter)


def Gaussian(image):
    filter = np.array([[1/16, 2/16, 1/16],
                       [2/16, 4/16, 2/16],
                       [1/16, 2/16, 1/16]])
    return filtering(image, filter)


def Prewitt(image):
    filter = np.array([[-1/6, 0, 1/6],
                       [-1/6, 0, 1/6],
                       [-1/6, 0, 1/6]])
    return filtering(image, filter)


def Sobel(image):
    filter = np.array([[-1/8, 0, 1/8],
                       [-2/8, 0, 2/8],
                       [-1/8, 0, 1/8]])
    return filtering(image, filter)


def Laplacian(image):
    filter = np.array([[1,  1, 1],
                       [1, -8, 1],
                       [1,  1, 1]])
    return filtering(image, filter)


def Sharpening(image):
    filter = np.array([[-1/16,    -1/16, -1/16],
                       [-1/16, 1+8*1/16, -1/16],
                       [-1/16,    -1/16, -1/16]])
    return filtering(image, filter)
