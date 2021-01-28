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
        self.size = self.ndarray.shape
        self.width = self.size[0]
        self.height = self.size[1]
        self.center = (self.width//2, self.height//2)
        self.color = True if len(self.size)==3 else False

    def fileopen(self, file=""):
        self.file = file
        if self.file == "":
            fTyp = [("", "*")]
            iDir = os.path.abspath(os.path.dirname(__file__))
            self.file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        self.ndarray = cv2.imread(self.file)
        self.measure()

    def filesave(self, file="tmp.png"):
        cv2.imwrite(file, self.ndarray)
