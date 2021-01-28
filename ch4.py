
import dip
#
import numpy as np
import cv2

x = np.arange(256)

def LineToneCurve(src, n=2.0):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    y = x * n
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def GammaToneCurve(src, gamma=2.0):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    y = 255 * (x/255) ** (1/gamma)
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def SToneCurveSin(src):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    y = 255 / 2 * (np.sin((x/255-1/2)*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def HistEqualize(src):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(src.ndarray.flatten(), 256, [0, 256])
    y = hist.cumsum()
    y_mask = np.ma.masked_equal(y, 0)
    y_mask = (y_mask-y_mask.min()) * 255 / (y_mask.max()-y_mask.min())
    y = np.ma.filled(y_mask, 0)
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def NegPosInversion(src):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    y = 255 - x
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def Solarization(src):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    y = 255 / 2 * (np.sin((x/255+1/2)*3*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def Posterization(src, L=6):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    if L < 2:
        L = 2
    elif L > 255:
        L = 255
    y = x // (256//L) * (255//(L-1))
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst


def Binarization(src, thrreshold=128):
    if src.color:
        src.ndarray = cv2.cvtColor(src.ndarray, cv2.COLOR_BGR2GRAY)
    y = np.zeros(x.shape)
    y[threshold<x] = 255
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Image()
    dst.ndarray = cv2.LUT(src.ndarray, y)
    return dst
