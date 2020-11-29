# cording: utf-8
"""
ディジタル画像処理GUI

@Author: Hayashi-
SpecialThanks: Sato kun
"""
import os
import cv2
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

np.seterr(divide="ignore")
os.chdir("C:/Users/TAKAHIRO/Pictures/PythonImages")

# --------------------------------------------------
# 画像 IMaGe:img

img_src = cv2.imread("src.png", 0)

wid = min(img_src.shape[0], img_src.shape[1])
img_src = img_src[0:wid, 0:wid]
size = img_src.shape

cv2.imwrite("crn.png", img_src)

center = wid//2
imgs = []

# --------------------------------------------------
# Shading

shd_txt = ["LineToneCurve", "GammaToneCurve", "SToneCurveSin",
           "NegPosInversion", "Solarization", "Posterization",
           "Binarization"]

x = np.arange(256)

def LineToneCurve():
    """折れ線トーンカーブ

    """
    n = 2.0
    y = x * n
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

def GammaToneCurve():
    gamma = 2.0
    y = 255 * (x/255) ** (1/gamma)
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

def SToneCurveSin():
    y = 255 / 2 * (np.sin((x/255-1/2)*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

def NegPosInversion():
    y = 255 - x
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

def Solarization():
    y = 255 / 2 * (np.sin((x/255+1/2)*3*np.pi)+1)
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

def Posterization():
    L = 6
    if L < 2:
        L = 2
    elif L > 255:
        L = 255
    y = x // (256//L) * (255//(L-1))
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

def Binarization():
    threshold = 128
    y = np.zeros(x.shape)
    y[threshold < x] = 255
    y = np.clip(y, 0, 255).astype(np.uint8)
    ShadingConversion(y)

# --------------------------------------------------
# フィルター　FiLTer:flt


flt_txt = ["Averaging", "Gaussian", "Prewitt", "Sobel", "Laplacian",
           "Sharpening", "LowPass", "HighPass", "BandPass", "HighEmphasis"]

def Averaging():
    flt_ary = np.array([["1/9", "1/9", "1/9"],
                        ["1/9", "1/9", "1/9"],
                        ["1/9", "1/9", "1/9"]], dtype=object)
    spatial_flt(flt_ary)

def Gaussian():
    flt_ary = np.array([["1/16", "2/16", "1/16"],
                        ["2/16", "4/16", "2/16"],
                        ["1/16", "2/16", "1/16"]], dtype=object)
    spatial_flt(flt_ary)

def Prewitt():
    flt_ary = np.array([["-1/6", "0", "1/6"],
                        ["-1/6", "0", "1/6"],
                        ["-1/6", "0", "1/6"]], dtype=object)
    spatial_flt(flt_ary)

def Sobel():
    flt_ary = np.array([["-1/8", "0", "1/8"],
                        ["-2/8", "0", "2/8"],
                        ["-1/8", "0", "1/8"]], dtype=object)
    spatial_flt(flt_ary)

def Laplacian():
    flt_ary = np.array([["1", " 1", "1"],
                        ["1", "-8", "1"],
                        ["1", " 1", "1"]], dtype=object)
    spatial_flt(flt_ary)

def Sharpening():
    flt_ary = np.array([["-1/16", "   -1/16", "-1/16"],
                        ["-1/16", "1+8*1/16", "-1/16"],
                        ["-1/16", "   -1/16", "-1/16"]], dtype=object)
    spatial_flt(flt_ary)

def LowPass():
    flt_mtr = np.full(size, 0)
    R = scl_R.get()
    for i in range(0, wid):
        for j in range(0, wid):
            if (i-center)*(i-center) + (j-center)*(j-center) < R*R:
                flt_mtr[i][j] = 1
    frequency_flt(flt_mtr)

def HighPass():
    flt_mtr = np.full(size, 1)
    R = scl_R.get()
    for i in range(0, wid):
        for j in range(0, wid):
            if (i-center)*(i-center) + (j-center)*(j-center) < R*R:
                flt_mtr[i][j] = 0
    frequency_flt(flt_mtr)

def BandPass():
    flt_mtr = np.full(size, 0)
    R = scl_R.get()
    r = scl_r.get()
    for i in range(0, wid):
        for j in range(0, wid):
            if (i-center)*(i-center) + (j-center)*(j-center) < R*R:
                flt_mtr[i][j] = 1
            if (i-center)*(i-center) + (j-center)*(j-center) < r*r:
                flt_mtr[i][j] = 0
    frequency_flt(flt_mtr)

def HighEmphasis():
    flt_mtr = np.full(size, 1)
    R = scl_R.get()
    for i in range(0, wid):
        for j in range(0, wid):
            if (i-center)*(i-center) + (j-center)*(j-center) > R*R:
                flt_mtr[i][j] += 1
    frequency_flt(flt_mtr)

# --------------------------------------------------
# 関数


def convert(img_gry):
    img_pil = Image.fromarray(img_gry)
    img_tk = ImageTk.PhotoImage(img_pil)
    imgs.append(img_tk)
    return img_tk


def ShadingConversion(tcurve):
    img_crn = cv2.imread("crn.png", 0)
    img_dst = cv2.LUT(img_crn, tcurve)
    cv2.imwrite("crn.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
    plt.figure()
    plt.subplots_adjust(left=0.14, right=0.9, bottom=0.14, top=0.91)
    plt.hist(img_crn.flatten(), bins=x, color="black")
    plt.xlim(0, 255)
    plt.yticks([])
    plt.savefig("crn_hst.png")
    plt.close()
    hst_crn = cv2.imread("crn_hst.png")
    hst_crn = cv2.resize(hst_crn, (wid//2, wid//2))
    cnv_spc.grid()
    cnv_spc.create_image(0, 0, image=convert(hst_crn), anchor="nw")
    plt.figure()
    plt.hist(img_dst.flatten(), bins=x, color="black")
    plt.subplots_adjust(left=0.14, right=0.9, bottom=0.14, top=0.91)
    plt.xlim(0, 255)
    plt.yticks([])
    plt.savefig("dst_hst.png")
    plt.close()
    hst_dst = cv2.imread("dst_hst.png")
    hst_dst = cv2.resize(hst_dst, (wid//2, wid//2))
    cnv_dst.grid()
    cnv_dst.create_image(0, 0, image=convert(hst_dst), anchor="nw")
    plt.figure()
    plt.subplots_adjust(left=0.14, right=0.9, bottom=0.14, top=0.91)
    plt.plot(x, tcurve, color="red")
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.savefig("crv.png")
    plt.close()
    img_crv = cv2.imread("crv.png")
    img_crv = cv2.resize(img_crv, (wid//2, wid//2))
    cnv_flt.grid()
    cnv_flt.create_image(0, 0, image=convert(img_crv), anchor="nw")


def spatial_flt(flt_ary):
    cnv_spc.grid_remove()
    frm_pxl.grid()
    frm_dst.grid()
    for i in range(9):
        lbl_flt[i]["text"] = flt_ary[i//3][i%3]
        flt_ary[i//3][i%3] = eval(flt_ary[i//3][i%3])
    flt_ary = flt_ary.astype(np.float64)
    frm_flt.grid()
    img_crn = cv2.imread("crn.png", 0)
    img_tmp = cv2.filter2D(img_crn, -1, flt_ary)
    img_dst = cv2.convertScaleAbs(img_tmp)
    cv2.imwrite("crn.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")


def frequency_flt(flt_mtr):
    plt.figure(dpi=1, figsize=(wid//2, wid//2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(flt_mtr, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("flt.png")
    plt.close()
    img_flt = cv2.imread("flt.png", 0)
    frm_prm.grid()
    cnv_flt.grid()
    cnv_flt.create_image(0, 0, image=convert(img_flt), anchor="nw")
    img_crn = cv2.imread("crn.png", 0)
    f = np.fft.fft2(img_crn)
    fshift = np.fft.fftshift(f)
    ftmp = fshift * flt_mtr
    funshift = np.fft.fftshift(ftmp)
    img_dst = np.uint8(np.fft.ifft2(funshift).real)
    cv2.imwrite("crn.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
    mag_spc = 20 * np.log(np.abs(ftmp))
    mag_spc[np.isinf(mag_spc)] = 0
    plt.figure(dpi=1, figsize=(wid//2, wid//2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(mag_spc, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("dst.png")
    plt.close()
    spc_dst = cv2.imread("dst.png", 0)
    cnv_dst.grid()
    cnv_dst.create_image(0, 0, image=convert(spc_dst), anchor="nw")

# --------------------------------------------------
# ボタン関数


def pointer(event):
    px = event.x
    py = event.y
    lbl_pxl[0]["text"] = img_src[px-1][py-1]
    lbl_pxl[1]["text"] = img_src[px][py-1]
    lbl_pxl[2]["text"] = img_src[px+1][py-1]
    lbl_pxl[3]["text"] = img_src[px-1][py]
    lbl_pxl[4]["text"] = img_src[px][py]
    lbl_pxl[5]["text"] = img_src[px+1][py]
    lbl_pxl[6]["text"] = img_src[px-1][py+1]
    lbl_pxl[7]["text"] = img_src[px][py+1]
    lbl_pxl[8]["text"] = img_src[px-1][py+1]
    img_crn = cv2.imread("crn.png", 0)
    lbl_dst[0]["text"] = img_crn[px-1][py-1]
    lbl_dst[1]["text"] = img_crn[px][py-1]
    lbl_dst[2]["text"] = img_crn[px+1][py-1]
    lbl_dst[3]["text"] = img_crn[px-1][py]
    lbl_dst[4]["text"] = img_crn[px][py]
    lbl_dst[5]["text"] = img_crn[px+1][py]
    lbl_dst[6]["text"] = img_crn[px-1][py+1]
    lbl_dst[7]["text"] = img_crn[px][py+1]
    lbl_dst[8]["text"] = img_crn[px-1][py+1]


def select(event):
    fTyp = [("", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    global img_src, wid, size, center
    img_src = cv2.imread(file, 0)
    wid = min(img_src.shape[0], img_src.shape[1])
    img_src = img_src[0:wid, 0:wid]
    size = img_src.shape
    cv2.imwrite("crn.png", img_src)
    center = wid//2
    cnv_img.configure(width=wid, height=wid)
    cnv_spc.configure(width=wid//2, height=wid//2)
    cnv_flt.configure(width=wid//2, height=wid//2)
    cnv_dst.configure(width=wid//2, height=wid//2)
    img_crn = cv2.imread("crn.png", 0)
    cnv_img.create_image(0, 0, image=convert(img_crn), anchor="nw")


def reset(event):
    cv2.imwrite("crn.png", img_src)
    img_crn = cv2.imread("crn.png", 0)
    cnv_img.create_image(0, 0, image=convert(img_crn), anchor="nw")


def shading(event):
    frm_pxl.grid_remove()
    cnv_flt.grid_remove()
    frm_prm.grid_remove()
    frm_flt.grid_remove()
    frm_dst.grid_remove()
    cnv_dst.grid_remove()
    eval(cmb_shd.get())()


def filtering(event):
    frm_pxl.grid_remove()
    cnv_flt.grid_remove()
    frm_prm.grid_remove()
    frm_flt.grid_remove()
    frm_dst.grid_remove()
    cnv_dst.grid_remove()
    eval(cmb_flt.get())()


def fourier(event):
    img_crn = cv2.imread("crn.png", 0)
    f = np.fft.fft2(img_crn)
    fshift = np.fft.fftshift(f)
    mag_spc = 20 * np.log(np.abs(fshift))
    plt.figure(dpi=1, figsize=(wid//2, wid//2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(mag_spc, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("spc.png")
    plt.close()
    img_spc = cv2.imread("spc.png", 0)
    cnv_spc.grid()
    cnv_spc.create_image(0, 0, image=convert(img_spc), anchor="nw")


# --------------------------------------------------
# GUI

root = tk.Tk()
root.title("DigitalImageProcessing GUI")

# Row=0
cnv_img = tk.Canvas(root, width=wid, height=wid)
cnv_img.grid(row=0, column=0, rowspan=6)
cnv_img.create_image(0, 0, image=convert(img_src), anchor="nw")
cnv_img.bind("<1>", pointer)
cnv_img.bind("<3>", select)

btn_rst = tk.Button(root, text="Reset")
btn_rst.grid(row=0, column=1, sticky=tk.W+tk.E)
btn_rst.bind("<1>", reset)

frm_pxl = tk.LabelFrame(root, text="Pixel Value 9*9", width=wid//2, height=wid//2)
frm_pxl.grid(row=0, column=2, rowspan=3)
frm_pxl.grid_propagate(False)
frm_pxl.grid_remove()

lbl_pxl = []
for i in range(9):
    lbl_pxl.append(tk.Label(frm_pxl, text="0", width=11, height=4))
    lbl_pxl[i].grid(row=i//3, column=i%3)

cnv_spc = tk.Canvas(root, width=wid//2, height=wid//2)
cnv_spc.grid(row=0, column=2, rowspan=3)
cnv_spc.grid_remove()

frm_flt = tk.LabelFrame(root, text="Selected Filter", width=wid//2, height=wid//2)
frm_flt.grid(row=0, column=3, rowspan=3)
frm_flt.grid_propagate(False)
frm_flt.grid_remove()

lbl_flt = []
for i in range(9):
    lbl_flt.append(tk.Label(frm_flt, text="0", width=11, height=4))
    lbl_flt[i].grid(row=i//3, column=i%3)

cnv_flt = tk.Canvas(root, width=wid//2, height=wid//2)
cnv_flt.grid(row=0, column=3, rowspan=3)
cnv_flt.grid_remove()

# Row=1
cmb_shd = ttk.Combobox(root, state="readonly", values=shd_txt)
cmb_shd.grid(row=1, column=1, sticky=tk.W+tk.E)
cmb_shd.set(shd_txt[0])

# Row=2
btn_shd = tk.Button(root, text="Shading Conversion")
btn_shd.grid(row=2, column=1, sticky=tk.W+tk.E)
btn_shd.bind("<1>", shading)

# Row=3
cmb_flt = ttk.Combobox(root, state="readonly", values=flt_txt)
cmb_flt.grid(row=3, column=1, sticky=tk.W+tk.E)
cmb_flt.set(flt_txt[0])

frm_prm = tk.LabelFrame(root, text="Filter Parameter", width=wid//2, height=wid//2)
frm_prm.grid(row=3, column=2, rowspan=3)
frm_prm.grid_propagate(False)
frm_prm.grid_remove()

lbl_R = tk.Label(frm_prm, text="R")
lbl_R.grid(row=0, column=0)

scl_R = tk.Scale(frm_prm, orient="horizontal", from_=0, to=255, length=230)
scl_R.grid(row=0, column=1, sticky=tk.E)
scl_R.set(50)

lbl_r = tk.Label(frm_prm, text="r")
lbl_r.grid(row=1, column=0)

scl_r = tk.Scale(frm_prm, orient="horizontal", from_=0, to=255, length=230)
scl_r.grid(row=1, column=1, sticky=tk.E)
scl_r.set(25)

frm_dst = tk.LabelFrame(root, text="Filtered Pixel Value 9*9", width=wid//2, height=wid//2)
frm_dst.grid(row=3, column=3, rowspan=3)
frm_dst.grid_propagate(False)
frm_dst.grid_remove()

lbl_dst = []
for i in range(9):
    lbl_dst.append(tk.Label(frm_dst, text="0", width=11, height=4))
    lbl_dst[i].grid(row=i//3, column=i%3)

cnv_dst = tk.Canvas(root, width=wid//2, height=wid//2)
cnv_dst.grid(row=3, column=3, rowspan=3)
cnv_dst.grid_remove()

# Row=4
btn_flt = tk.Button(root, text="Filtering")
btn_flt.grid(row=4, column=1, sticky=tk.W+tk.E)
btn_flt.bind("<1>", filtering)

# Row=5
btn_fourier = tk.Button(root, text="Fourier Transform")
btn_fourier.grid(row=5, column=1, sticky=tk.W+tk.E)
btn_fourier.bind("<1>", fourier)

root.mainloop()
