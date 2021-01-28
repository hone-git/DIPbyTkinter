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

class Image:
    def __init__(self):
        self.ndarray = np.full((32, 32), 0)
        self.measure()

    def measure(self):
        self.size = (self.ndarray.shape[0], self.ndarray.shape[1])
        self.width = self.size[0]
        self.height = self.size[1]
        self.center = (self.width//2, self.height//2)
        self.color = True if self.ndarray.shape[-1]==3 else False

    def replace(self, ndarray):
        self.ndarray = ndarray
        self.measure()

    def fileopen(self, file=""):
        self.file = file
        if self.file == "":
            fTyp = [("", "*")]
            iDir = os.path.abspath(os.path.dirname(__file__))
            self.file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        self.replace(cv2.imread(self.file))

    def filesave(self, file="tmp.png"):
        cv2.imwrite(file, self.ndarray)

    def fft(self):
        if self.color:
            self.ndarray = cv2.cvtColor(self.ndarray, cv2.COLOR_BGR2GRAY)
        self.f = np.fft.fft2(self.ndarray)
        self.fshift = np.fft.fftshift(self.f)
        self.spectrum = 20 * np.log(np.abs(self.f))
        self.spectrum[np.isinf(self.spectrum)] = 0
