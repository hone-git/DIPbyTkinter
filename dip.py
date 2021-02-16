# ファイル関係
import os
from os.path import expanduser
from pathlib import Path
# 画像処理関係
import numpy as np
import cv2
from PIL import Image, ImageTk
from scipy import signal
# GUI関係
import tkinter as tk
import tkinter.filedialog

class Imeji:
    def __init__(self, img=None):
        if type(img) == str:
            self.openfile(img)
        elif type(img) == np.ndarray:
            if img.shape[-1] == 3:
                self.__color = img
                self.__gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                self.__gray = img
        else:
            self.openfile()
        self.measure()

    @property
    def gray(self):
        return self.__gray

    @property
    def color(self):
        try:
            return self.__color
        except AttributeError:
            print("This Imeji don't has color. return gray")
            return self.__gray

    @property
    def r(self):
        try:
            return self.__r
        except AttributeError:
            self.r, self.g, self.b = cv2.split(self.__color)
            return self.__r

    @property
    def g(self):
        try:
            return self.__g
        except AttributeError:
            self.r, self.g, self.b = cv2.split(self.__color)
            return self.__g

    @property
    def b(self):
        try:
            return self.__b
        except AttributeError:
            self.r, self.g, self.b = cv2.split(self.__color)
            return self.__b

    @property
    def hsv(self):
        try:
            return self.__hsv
        except AttributeError:
            self.__hsv = cv2.cvtColor(self.__color, cv2.COLOR_RGB2HSV)
            return self.__hsv

    @property
    def h(self):
        try:
            return self.__h
        except AttributeError:
            self.h, self.s, self.v = cv2.split(self.__hsv)
            return self.__h

    @property
    def s(self):
        try:
            return self.__s
        except AttributeError:
            self.h, self.s, self.v = cv2.split(self.__hsv)
            return self.__s

    @property
    def v(self):
        try:
            return self.__v
        except AttributeError:
            self.h, self.s, self.v = cv2.split(self.__hsv)
            return self.__v

    @property
    def pil(self):
        try:
            return self.__pil
        except AttributeError:
            self.__pil = Image.fromarray(self.__color)
            return self.__pil

    @property
    def tk(self):
        try:
            return self.__tk
        except AttributeError:
            try:
                self.__tk = ImageTk.PhotoImage(self.__pil)
            except AttributeError:
                self.__pil = Image.fromarray(self.__color)
                self.__tk = ImageTk.PhotoImage(self.__pil)
            return self.__tk

    @property
    def f(self):
        try:
            return self.__f
        except AttributeError:
            self.__f = np.fft.fft2(self.__color)
            return self.__f

    @property
    def fshift(self):
        try:
            return self.__fshift
        except AttributeError:
            self.fft()
            return self.__fshift

    def openfile(self, file=""):
        self.file = file
        if self.file == "":
            fTyp = [("", "*")]
            iDir = os.path.abspath(os.path.dirname(__file__))
            self.file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        tmp = cv2.imread(self.file)
        self.__color = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        self.__gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    def savefile(self, file="tmp.png"):
        tmp = cv2.cvtColor(self.__color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file, tmp)

    def measure(self):
        self.height = self.__gray.shape[0]
        self.width = self.__gray.shape[1]
        self.size = (self.height, self.width)
        self.center = (self.height//2, self.width//2)

    def fft(self):
        self.__f = np.fft.fft2(self.__color)
        self.__fshift = np.fft.fftshift(self.__f)
        self.spectrum = 20 * np.log(np.abs(self.__f))
        self.spectrum[np.isinf(self.spectrum)] = 0


class Shori:
    def __init__(self, name, process, params={}):
        self.name = name
        self.process = process
        self.params = params

if __name__ == '__main__':
    first = Imeji()

    wie = signal.wiener(first.gray).astype(np.uint8)
    print(wie)
    cv2.imshow("test", wie)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
