# ファイル関係
import os
from os.path import expanduser
from pathlib import Path
# 画像処理関係
import numpy as np
import cv2
from PIL import Image, ImageTk
# GUI関係
import tkinter as tk
import tkinter.filedialog

class Imeji:
    def __init__(self, img=None):
        if type(img) == str:
            self.openfile(img)
        elif type(img) == np.ndarray:
            self.__color = img
            if img.shape[-1] == 3:
                self.__gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                self.__gray = img
        else:
            self.openfile()
        self.measure()

    @property
    def color(self):
        return self.__color

    @property
    def gray(self):
        return self.__gray

    @property
    def tk(self):
        try:
            return self.__tk
        except AttributeError:
            self.convertForm()
            return self.__tk

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
        self.height = self.__color.shape[0]
        self.width = self.__color.shape[1]
        self.size = (self.height, self.width)
        self.center = (self.height//2, self.width//2)

    def convertColor(self):
        self.r, self.g, self.b = cv2.split(self.__color)

    def convertForm(self):
        self.__pil = Image.fromarray(self.__color)
        self.__tk = ImageTk.PhotoImage(self.__pil)

    def fft(self):
        self.__f = np.fft.fft2(self.__color)
        self.__fshift = np.fft.fftshift(self.__f)
        self.spectrum = 20 * np.log(np.abs(self.__f))
        self.spectrum[np.isinf(self.spectrum)] = 0


if __name__ == '__main__':
    first = Imeji()
    print(first.gray)
