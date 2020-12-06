# cording: utf-8
"""
! ファイルの書き込み場所は
  C:/Users/****/Pictures/dipImage
"""
import os
from os.path import expanduser
import cv2
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

np.seterr(divide="ignore")

plt.rcParams["xtick.major.bottom"] = False
plt.rcParams["ytick.major.left"] = False
plt.rcParams["figure.subplot.left"] = 0
plt.rcParams["figure.subplot.bottom"] = 0
plt.rcParams["figure.subplot.right"] = 1
plt.rcParams["figure.subplot.top"] = 1

dir = expanduser("~") + "/Pictures/dipImage"
try:
    os.chdir(dir)
except FileNotFoundError:
    os.mkdir(dir)
    os.chdir(dir)

# --------------------------------------------------
# 画像 IMaGe:img
imgs = []

# --------------------------------------------------
# Shading

shd_txt = ["LineToneCurve", "GammaToneCurve", "SToneCurveSin",
           "NegPosInversion", "Solarization", "Posterization","Binarization"]

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


flt_txt = ["Averaging", "Gaussian", "Prewitt", "Sobel", "Laplacian", "Sharpening",
           "LowPass", "HighPass", "BandPass", "HighEmphasis",
           "GaussianLowPass", "GaussianHighPass", "GaussianHighEmphasis"]

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

def GaussianLowPass():
    flt_mtr = Gaussian2D(wid)
    frequency_flt(flt_mtr)

def GaussianHighPass():
    flt_mtr = Gaussian2D(wid)*(-1)+1
    frequency_flt(flt_mtr)

def GaussianHighEmphasis():
    flt_mtr = Gaussian2D(wid)*(-1)+2
    frequency_flt(flt_mtr)

# --------------------------------------------------
# 関数


def convert(img_gry):
    img_pil = Image.fromarray(img_gry)
    img_tk = ImageTk.PhotoImage(img_pil)
    imgs.append(img_tk)
    return img_tk


def replace():
    frame_process.grid()
    img_src = cv2.imread("src.png")
    img_dst = cv2.imread("dst.png")
    img_operator = cv2.imread("operator.png")
    img_src = cv2.resize(img_src, (wid//2, wid//2))
    img_dst = cv2.resize(img_dst, (wid//2, wid//2))
    img_operator = cv2.resize(img_operator, (wid//2, wid//2))
    canvas_src.create_image(0, 0, image=convert(img_src), anchor="nw")
    canvas_dst.create_image(0, 0, image=convert(img_dst), anchor="nw")
    canvas_operator.create_image(0, 0, image=convert(img_operator), anchor="nw")


def ShadingConversion(tcurve):
    img_src = cv2.imread("img.png", 0)
    img_dst = cv2.LUT(img_src, tcurve)
    cv2.imwrite("img.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
    plt.figure()
    plt.hist(img_src.flatten(), bins=x, color="black")
    plt.xlim(0, 255)
    plt.savefig("src.png")
    plt.close()
    plt.figure()
    plt.hist(img_dst.flatten(), bins=x, color="black")
    plt.xlim(0, 255)
    plt.savefig("dst.png")
    plt.close()
    plt.figure()
    plt.subplots_adjust(left=0.14, right=0.9, bottom=0.14, top=0.91)
    plt.plot(x, tcurve, color="red")
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.savefig("operator.png")
    plt.close()
    replace()


def spatial_flt(flt_ary):
    fig,ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=flt_ary, cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    cell_height = 1 / len(flt_ary)
    for pos, cell in tb.get_celld().items():
        cell.set_height(cell_height)
    plt.savefig("operator.png")
    plt.close()
    for i in range(9):
        flt_ary[i//3][i%3] = eval(flt_ary[i//3][i%3])
    flt_ary = flt_ary.astype(np.float64)
    img_src = cv2.imread("img.png", 0)
    img_tmp = cv2.filter2D(img_src, -1, flt_ary)
    img_dst = cv2.convertScaleAbs(img_tmp)
    cv2.imwrite("img.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
    cv2.imwrite("src.png", img_src)
    cv2.imwrite("dst.png", img_dst)
    PixelValue()
    replace()


def frequency_flt(flt_mtr):
    cv2.imwrite("operator.png", flt_mtr*256)
    img_src = cv2.imread("img.png", 0)
    f = np.fft.fft2(img_src)
    fshift = np.fft.fftshift(f)
    ftmp = fshift * flt_mtr
    funshift = np.fft.fftshift(ftmp)
    img_dst = np.uint8(np.fft.ifft2(funshift).real)
    cv2.imwrite("img.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
    spc_src = 20 * np.log(np.abs(fshift))
    spc_src[np.isinf(spc_src)] = 0
    plt.figure(dpi=1, figsize=(wid//2, wid//2))
    plt.imshow(spc_src, cmap="gray")
    plt.savefig("src.png")
    plt.close()
    spc_dst = 20 * np.log(np.abs(ftmp))
    spc_dst[np.isinf(spc_dst)] = 0
    plt.figure(dpi=1, figsize=(wid//2, wid//2))
    plt.imshow(spc_dst, cmap="gray")
    plt.savefig("dst.png")
    plt.close()
    replace()


def FileSelect():
    fTyp = [("", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    global img_god, wid, size, center
    img_god = cv2.imread(file, 0)
    wid = min(img_god.shape[0], img_god.shape[1])
    img_god = img_god[0:wid, 0:wid]
    size = img_god.shape
    cv2.imwrite("god.png", img_god)
    cv2.imwrite("img.png", img_god)
    center = wid//2
    WidgetSize(wid)


def WidgetSize(wid):
    cnv_img.configure(width=wid, height=wid)
    frame_buttons.configure(width=120, height=wid)
    canvas_src.configure(width=wid//2, height=wid//2)
    canvas_dst.configure(width=wid//2, height=wid//2)
    canvas_operator.configure(width=wid//2, height=wid//2)
    frame_parameter.configure(width=wid//2, height=wid//2)

def PixelValue(x=16, y=16):
    img_src = cv2.imread("god.png", 0)
    img_dst = cv2.imread("img.png", 0)
    src = ([[img_src[x-1][y-1], img_src[x][y-1], img_src[x+1][y-1]],
            [img_src[x-1][y  ], img_src[x][y  ], img_src[x+1][y  ]],
            [img_src[x-1][y+1], img_src[x][y+1], img_src[x+1][y+1]]])
    fig,ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=src, cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    cell_height = 1 / len(src)
    for pos, cell in tb.get_celld().items():
        cell.set_height(cell_height)
    plt.savefig("src.png")
    plt.close()
    dst = ([[img_dst[x-1][y-1], img_dst[x][y-1], img_dst[x+1][y-1]],
            [img_dst[x-1][y  ], img_dst[x][y  ], img_dst[x+1][y  ]],
            [img_dst[x-1][y+1], img_dst[x][y+1], img_dst[x+1][y+1]]])
    fig,ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=dst, cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    cell_height = 1 / len(dst)
    for pos, cell in tb.get_celld().items():
        cell.set_height(cell_height)
    plt.savefig("dst.png")
    plt.close()

def Gaussian2D(width, sigma=4):
    x = np.linspace(-15, 15, width)
    y = np.linspace(-15, 15, width)
    matrix = np.full((width, width), 0, dtype=np.float64)
    for i in range(width):
        for j in range(width):
            matrix[i][j] = 1 / (2*np.pi*sigma**2)*np.exp(-1*(x[i]**2+y[j]**2)/(2*sigma**2))
    matrix = np.clip(matrix/matrix[width//2][width//2], 0, 1)
    return matrix

# --------------------------------------------------
# ボタン関数


def pointer(event):
    px = event.x
    py = event.y
    PixelValue(px, py)
    replace()


def select(event):
    FileSelect()
    img = cv2.imread("img.png", 0)
    cnv_img.create_image(0, 0, image=convert(img), anchor="nw")


def reset(event):
    cv2.imwrite("img.png", img_god)
    img = cv2.imread("img.png", 0)
    cnv_img.create_image(0, 0, image=convert(img), anchor="nw")


def shading(event):
    eval(cmb_shd.get())()


def filtering(event):
    eval(cmb_flt.get())()


def fourier(event):
    img = cv2.imread("img.png", 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag_spc = 20 * np.log(np.abs(fshift))
    plt.figure(dpi=1, figsize=(wid//2, wid//2))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(mag_spc, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("src.png")
    plt.close()
    img_src = cv2.imread("src.png", 0)
    cnv_spc.grid()
    cnv_spc.create_image(0, 0, image=convert(img_src), anchor="nw")


# --------------------------------------------------
# GUI

root = tk.Tk()
root.title("DigitalImageProcessing GUI")

cnv_img = tk.Canvas(root)
cnv_img.grid(row=0, column=0)
cnv_img.bind("<1>", pointer)
cnv_img.bind("<3>", select)

# Processing Panel
frame_buttons = tk.LabelFrame(root, text="Proccesing Panel")
frame_buttons.grid(row=0, column=1)
frame_buttons.grid_propagate(False)

btn_rst = tk.Button(frame_buttons, text="Reset")
btn_rst.grid(row=0, column=0, sticky=tk.E+tk.W)
btn_rst.bind("<1>", reset)

cmb_shd = ttk.Combobox(frame_buttons, state="readonly", values=shd_txt)
cmb_shd.grid(row=1, column=0, sticky=tk.E+tk.W)
cmb_shd.set(shd_txt[0])

btn_shd = tk.Button(frame_buttons, text="Shading Conversion")
btn_shd.grid(row=2, column=0, sticky=tk.E+tk.W)
btn_shd.bind("<1>", shading)

cmb_flt = ttk.Combobox(frame_buttons, state="readonly", values=flt_txt)
cmb_flt.grid(row=3, column=0, sticky=tk.E+tk.W)
cmb_flt.set(flt_txt[0])

btn_flt = tk.Button(frame_buttons, text="Filtering")
btn_flt.grid(row=4, column=0, sticky=tk.E+tk.W)
btn_flt.bind("<1>", filtering)

btn_fourier = tk.Button(frame_buttons, text="Fourier Transform")
btn_fourier.grid(row=5, column=0, sticky=tk.E+tk.W)
btn_fourier.bind("<1>", fourier)

for i, child in enumerate(frame_buttons.winfo_children()):
    frame_buttons.grid_rowconfigure(i, weight=1)

frame_buttons.grid_columnconfigure(0, weight=1)

# Processing Detail
frame_process = tk.LabelFrame(root, text="Processing Detail")
frame_process.grid(row=0, column=5)
frame_process.grid_remove()

canvas_src = tk.Canvas(frame_process)
canvas_src.grid(row=0, column=0)

canvas_operator = tk.Canvas(frame_process)
canvas_operator.grid(row=0, column=1)

frame_parameter = tk.LabelFrame(frame_process, text="Proccesing Parameter")
frame_parameter.grid(row=1, column=0)

lbl_R = tk.Label(frame_parameter, text="R")

scl_R = tk.Scale(frame_parameter, orient="horizontal", from_=0, to=255, length=230)
scl_R.set(50)

lbl_r = tk.Label(frame_parameter, text="r")

scl_r = tk.Scale(frame_parameter, orient="horizontal", from_=0, to=255)
scl_r.set(25)

canvas_dst = tk.Canvas(frame_process)
canvas_dst.grid(row=1, column=1)

# --------------------------------------------------
# main
FileSelect()
cnv_img.create_image(0, 0, image=convert(img_god), anchor="nw")

root.mainloop()
