import dip
#
import numpy as np
import cv2

#processes = ['LineToneCurve', 'GammaToneCurve',
#             'SToneCurveSin', 'HistEqualize', 'NegPosInversion',
#             'Solarization', 'Posterization', 'Binarization',
#             'SolarizationColor', 'LineToneCurveRGB', 'PseudoColor',
#             'LineToneCurveHSV', 'Emboss', 'AlphaBlending']

processes = []

x = np.arange(256)

def ShadingConversion(src, tonecurve):
    dst = dip.Imeji(cv2.LUT(src.gray, tonecurve))
    return dst


def LineToneCurve(image, n=2.0):
    y = x * n
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)

processes.append(dip.Shori("LineToneCurve", LineToneCurve, {"n":(0, 4.0, 0.1)}))

def GammaToneCurve(image, gamma=2.0):
    y = 255 * (x/255) ** (1/gamma)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def SToneCurveSin(image):
    y = 255 / 2 * (np.sin((x/255-1/2)*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def HistEqualize(image):
    hist, _ = np.histogram(image.cv2.flatten(), 256, [0, 256])
    y = hist.cumsum()
    y_mask = np.ma.masked_equal(y, 0)
    y_mask = (y_mask-y_mask.min()) * 255 / (y_mask.max()-y_mask.min())
    y = np.ma.filled(y_mask, 0)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def NegPosInversion(image):
    y = 255 - x
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def Solarization(image):
    y = 255 / 2 * (np.sin((x/255+1/2)*3*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def Posterization(image, L=6):
    if L < 2:
        L = 2
    elif L > 255:
        L = 255
    y = x // (256//L) * (255//(L-1))
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def Binarization(image, thrreshold=128):
    y = np.zeros(x.shape)
    y[threshold<x] = 255
    y = np.clip(y, 0, 255).astype(np.uint8)
    return ShadingConversion(image, y)


def SolarizationColor(image):
    y = 255 / 2 * (np.sin((x/255-1/2)*9*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    dst = dip.Imeji(cv2.LUT(src.color, y))
    return dst


def LineToneCurveRGB(image, target='rgb', n=2.0):
    y = x * n
    y = np.clip(y, 0, 255).astype(np.uint8)
    b, g, r = image.b, image.g, image.r
    if target in 'R' or target in 'r':
        r = cv2.LUT(r, y)
    if target in 'G' or target in 'g':
        r = cv2.LUT(g, y)
    if target in 'B' or target in 'b':
        r = cv2.LUT(b, y)
    dst = dip.Imeji(cv2.merge([b, g, r]))
    return dst


def PseudoColor(image):
    y_b = -4 * x + 512
    y_g = x.copy()
    y_g[:128] = 4 * x[:128]
    y_g[128:] = -4 * x[128:] + 1024
    y_r = 4 * x - 512
    y_b = np.clip(y_b, 0, 255).astype(np.uint8)
    y_g = np.clip(y_g, 0, 255).astype(np.uint8)
    y_r = np.clip(y_r, 0, 255).astype(np.uint8)
    b = cv2.LUT(image.gray, y_b)
    g = cv2.LUT(image.gray, y_g)
    r = cv2.LUT(image.gray, y_r)
    dst = dip.Imeji(cv2.merge([b, g, r]))
    return dst


def LineToneCurveHSV(image, target='hsv', n=2.0):
    y = x * n
    y = np.clip(y, 0, 255).astype(np.uint8)
    h, s, v = image.h, image.s, image.v
    if target in 'H' or target in 'h':
        h = cv2.LUT(h, y)
    if target in 'S' or target in 's':
        s = cv2.LUT(s, y)
    if target in 'V' or target in 'v':
        v = cv2.LUT(v, y)
    dst = dip.Imeji(cv2.merge([h, s, v]))
    return dst


def Emboss(image, shift=(20, 20)):
    y = 255 - x
    y = np.clip(y, 0, 255).astype(np.uint8)
    inverted = cv2.LUT(image, y)
    mat = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    shifted = cv2.warpAffine(inverted, mat, (inverted.shape[1], inverted.shape[0]))
    dst = dip.Imeji(np.clip(image.gray+shifted-128, 0, 255).astype(np.uint8))
    return dst


def AlphaBlending(image1, image2, alpha=0.5):
    if alpha < 0:
        alpha = 0.0
    elif alpha > 1:
        alpha = 1.0
    blend = cv2.addWeighted(image1, alpha, image2, 1-alpha, 0)
    dst = dip.Imeji(np.clip(blend, 0, 255).astype(np.uint8))
    return dst
