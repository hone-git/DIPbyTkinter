import dip
#
import numpy as np
import cv2

processes = ['Averaging', 'Gaussian', 'DirectionAveraging',
             'Derivative', 'Prewitt', 'Sobel', 'Laplacian', 'Sharpening',
             'kNNAveraging', 'Bilateral', 'NLMean', 'Median']

def SpatialFiltering(src, filter):
    tmp = cv2.filter2D(src.color, -1, filter)
    dst = dip.Imeji(cv2.convertScaleAbs(tmp))
    return dst


def Averaging(image, scale=3):
    if scale%2 == 0:
        scale = 3
    filter = np.full((scale, scale), 1/scale**2, dtype=np.float64)
    return SpatialFiltering(image, filter)


def Gaussian(image, scale=3):
    if scale%2 != 1:
        scale = 3
    def pascal(n):
        if n<1 :
            return [1]
        p = pascal(n-1)
        return list(map(lambda a,b:a+b, [0]+p, p+[0]))
    p = pascal(scale-1)
    filter = np.full((scale, scale), 0, dtype=np.float64)
    for i in range(scale):
        for j in range(scale):
            filter[i][j] = p[i] * p[j]
    filter = np.clip(filter/filter.sum(), 0, 1)
    return SpatialFiltering(image, filter)


def DirectionAveraging(image, start='NW', end='SE', scale=3):
    if scale%2 != 1:
        scale = 3
    filter = np.full((scale, scale), 0, dtype=np.float64)
    d = {'N':(0, (scale-1)/2), 'S':(scale-1, (scale-1)/2),
         'W':((scale-1)/2, 0), 'E':((scale-1)/2, scale-1),
         'NW':(0, 0), 'NE':(0, scale-1), 'SW':(scale-1, 0), 'SE':(scale-1, scale-1)}
    x = np.linspace(d[start][0], d[end][0], scale)
    y = np.linspace(d[start][1], d[end][1], scale)
    for i in range(scale):
        filter[int(x[i])][int(y[i])] = 1
    filter = np.clip(filter/filter.sum(), 0, 1)
    return SpatialFiltering(image, filter)


def Derivative(image, direction='right'):
    if direction == 'right':
        filter = np.array([[0,    0,   0],
                           [0, -1/2, 1/2],
                           [0,    0,   0]])
    elif direction == 'left':
        filter = np.array([[  0,    0, 0],
                           [1/2, -1/2, 0],
                           [  0,    0, 0]])
    elif direction == 'up':
        filter = np.array([[0,  1/2, 0],
                           [0, -1/2, 0],
                           [0,    0, 0]])
    elif direction == 'down':
        filter = np.array([[0,    0, 0],
                           [0, -1/2, 0],
                           [0,  1/2, 0]])
    return SpatialFiltering(image, filter)


def Prewitt(image, direction='horizontal'):
    if direction == 'horizontal':
        filter = np.array([[-1/6, 0, 1/6],
                           [-1/6, 0, 1/6],
                           [-1/6, 0, 1/6]])
    elif direction == 'vertical':
        filter = np.array([[ 1/6,  1/6,  1/6],
                           [   0,    0,    0],
                           [-1/6, -1/6, -1/6]])
    return SpatialFiltering(image, filter)


def Sobel(image, direction='horizontal'):
    if direction == 'horizontal':
        filter = np.array([[-1/6, 0, 1/6],
                           [-2/8, 0, 1/8],
                           [-1/6, 0, 1/6]])
    elif direction == 'vertical':
        filter = np.array([[ 1/8,  2/8,  1/8],
                           [   0,    0,    0],
                           [-1/8, -2/8, -1/8]])
    return SpatialFiltering(image, filter)


def Laplacian(image, direction=8):
    if direction == 4:
        filter = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]])
    elif direction == 8:
        filter = np.array([[1,  1, 1],
                           [1, -8, 1],
                           [1,  1, 1]])
    return SpatialFiltering(image, filter)


def Sharpening(image, k=9):
    filter = np.array([[-k/9,    -k/9, -k/9],
                       [-k/9, 1+8*k/9, -k/9],
                       [-k/9,    -k/9, -k/9]])
    return SpatialFiltering(image, filter)


def kNNAveraging(image, k=3, scale=3):
    if scale%2 != 1:
        scale = 3
    if k > scale**2 / 2:
        k = scale**2 // 2
    scale = (scale-1)//2
    img = np.full(image.size, 0, dtype=np.float64)
    for i in range(image.height):
        for j in range(image.width):
            f = image.gray[i-scale:i+scale+1, j-scale:j+scale+1].flatten()
            dif = f - image.gray[i][j]
            f = f[np.argsort(dif)]
            try:
                img[i][j] = f[0:k].sum() // k
            except IndexError:
                img[i][j] = image.gray[i][j]
    dst = dip.Imeji(img)
    return dst


def Bilateral(image, d=15, sigmaColor=50, sigmaSpace=50):
    dst = dip.Imeji(cv2.bilateralFilter(image.color, d, sigmaColor, sigmaSpace))
    return dst


def NLMean(image, h=10, hColor=10, templateWS=7, searchWS=21):
    dst = dip.Imeji(cv2.fastNlMeansDenoisingColored(image.color, None, h, hColor, templateWS, searchWS))
    return dst


def Median(image, scale=3):
    dst = dip.Imeji(cv2.medianBlur(image.color, scale))
    return dst
