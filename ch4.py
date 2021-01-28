import dip
#
import numpy as np
import cv2

x = np.arange(256)

def ShadingConversion(src, tonecurve):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    dst = dip.Image()
    dst.replace(cv2.LUT(src.ndarray, tonecurve))
    return dst


def LineToneCurve(image, n=2.0):
    y = x * n
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def GammaToneCurve(image, gamma=2.0):
    y = 255 * (x/255) ** (1/gamma)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def SToneCurveSin(image):
    y = 255 / 2 * (np.sin((x/255-1/2)*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def HistEqualize(image):
    hist, _ = np.histogram(image.ndarray.flatten(), 256, [0, 256])
    y = hist.cumsum()
    y_mask = np.ma.masked_equal(y, 0)
    y_mask = (y_mask-y_mask.min()) * 255 / (y_mask.max()-y_mask.min())
    y = np.ma.filled(y_mask, 0)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def NegPosInversion(image):
    y = 255 - x
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def Solarization(image):
    y = 255 / 2 * (np.sin((x/255+1/2)*3*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def Posterization(image, L=6):
    if L < 2:
        L = 2
    elif L > 255:
        L = 255
    y = x // (256//L) * (255//(L-1))
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)


def Binarization(image, thrreshold=128):
    y = np.zeros(x.shape)
    y[threshold<x] = 255
    y = np.clip(y, 0, 255).astype(np.uint8)
    return Conversion(image, y)
