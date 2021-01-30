import dip
#
import numpy as np
import cv2

def GaussianDist(size, sigma):
    x = np.linspace(-16, 16, size[0])
    y = np.linspace(-16, 16, size[1])
    distribution = np.full(size, 0, dtype=np.float64)
    for i in range(size[0]):
        for j in range(size[1]):
            distribution[i][j] = 1 / (2*np.pi*sigma**2)*np.exp(-1*(x[i]**2+y[j]**2)/(2*sigma**2))
    distribution = np.clip(distribution/distribution[size[0]//2][size[1]//2], 0, 1)
    return distribution


def FrequencyFiltering(src, filter):
    ftmp = src.fshift * filter
    funshift = np.fft.fftshift(ftmp)
    dst = dip.Imeji(np.uint8(np.fft.ifft2(funshift).real))
    return dst


def LowPass(image, R=50):
    filter = np.full(image.size, 0)
    for i in range(0, image.width):
        for j in range(0, image.height):
            if (i-image.center[0])**2 + (j-image.center[1])**2 < R*R:
                filter[i][j] = 1
    return FrequencyFiltering(image, filter)


def HighPass(image, R=50):
    filter = np.full(image.size, 1)
    for i in range(0, image.width):
        for j in range(0, image.height):
            if (i-image.center[0])**2 + (j-image.center[1])**2 < R*R:
                filter[i][j] = 0
    return FrequencyFiltering(image, filter)


def BandPass(image, R=75, r=25):
    filter = np.full(image.size, 0)
    for i in range(0, image.width):
        for j in range(0, image.height):
            if (i-image.center[0])**2 + (j-image.center[1])**2 < R*R:
                filter[i][j] = 1
            if (i-image.center[0])**2 + (j-image.center[1])**2 < r*r:
                filter[i][j] = 0
    return FrequencyFiltering(image, filter)


def HighEmphasis(image, R=50):
    filter = np.full(image.size, 1)
    for i in range(0, image.width):
        for j in range(0, image.height):
            if (i-image.center[0])**2 + (j-image.center[1])**2 < R*R:
                filter[i][j] += 1
    return FrequencyFiltering(image, filter)


def GaussianLowPass(image, sigma=4):
    filter = GaussianDist(image.size, sigma)
    return FrequencyFiltering(image, filter)


def GaussianHighPass(image, sigma=4):
    filter = GaussianDist(image.size, sigma)*(-1)+1
    return FrequencyFiltering(image, filter)


def GaussianBandPass(image, sigma=4, Dsigma=3.8):
    filter = GaussianDist(image.size, sigma) - GaussianDist(image.size, Dsigma)
    return FrequencyFiltering(image, filter)


def GaussianHighEmphasis(image, sigma=4, rate=1):
    filter = rate + 1 - GaussianDist(image.size, sigma) * rate
    return FrequencyFiltering(image, filter)
