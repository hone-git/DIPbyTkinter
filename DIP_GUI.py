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
plt.rcParams["font.size"] = 16

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
           "GaussianLowPass", "GaussianHighPass", "GaussianBandPass", "GaussianHighEmphasis",
           "Resize"]

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
    sigma = scl_sigma.get()
    flt_mtr = Gaussian2D(wid, sigma)
    frequency_flt(flt_mtr)

def GaussianHighPass():
    sigma = scl_sigma.get()
    flt_mtr = Gaussian2D(wid, sigma)*(-1)+1
    frequency_flt(flt_mtr)

def GaussianBandPass():
    sigma = scl_sigma.get()
    Dsigma = scl_Dsigma.get()
    flt_mtr = Gaussian2D(wid, sigma) - Gaussian2D(wid, Dsigma)
    frequency_flt(flt_mtr)

def GaussianHighEmphasis():
    sigma = scl_sigma.get()
    rate = 1
    flt_mtr = rate + 1 - Gaussian2D(wid, sigma) * rate
    frequency_flt(flt_mtr)

def Resize():
    global wid, size, center
    rate = scl_rate.get()
    img_src = cv2.imread("img.png", 0)
    f = np.fft.fft2(img_src)
    fshift = np.fft.fftshift(f)
    ftmp = np.full((int(wid*rate), int(wid*rate)), 0, dtype=np.complex128)
    ftmp[int(wid*(rate-1)/2):int(wid*(rate+1)/2), int(wid*(rate-1)/2):int(wid*(rate+1)/2)] = fshift
    funshift = np.fft.fftshift(ftmp)
    img_dst = np.uint8(np.fft.ifft2(funshift).real)
    img_dst = img_dst * (rate**2)
    cv2.imwrite("img.png", img_dst)
    wid = int(wid * rate)
    size = img_dst.shape
    center = wid*rate//2
    WidgetSize(wid)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")

# --------------------------------------------------
# グラフ描画
def Histgram(input, output):
    """ヒストグラム描画/保存

    input:入力画像
    output:出力画像[ファイル名]
    """
    plt.figure()
    plt.hist(input.flatten(), bins=x, color="black")
    plt.xlim(0, 255)
    plt.savefig(output+".png")
    plt.close()


def Table(input, output):
    """表描画/保存

    input:入力行列
    output:出力画像[ファイル名]
    """
    fig,ax = plt.subplots()
    ax.axis('off')
    tb = ax.table(cellText=input, cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    cell_height = 1 / len(input)
    for pos, cell in tb.get_celld().items():
        cell.set_height(cell_height)
    plt.savefig(output+".png")
    plt.close()


def Spectrum(input, output, isSpc=False):
    """スペクトラム描画/保存

    input:入力画像
    output:出力画像[ファイル名]
    isSpc:スペクトラム判定
    """
    if not(isSpc):
        input = np.fft.fft2(input)
        input = np.fft.fftshift(input)
    spc = 20 * np.log(np.abs(input))
    spc[np.isinf(spc)] = 0
    spc = spc / DC_amp * 256
    cv2.imwrite(output+".png", spc)


# --------------------------------------------------
# 関数


def convert(img_gry):
    img_pil = Image.fromarray(img_gry)
    img_tk = ImageTk.PhotoImage(img_pil)
    imgs.append(img_tk)
    return img_tk


def replace():
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
    plt.subplots_adjust(left=0.14, right=0.9, bottom=0.14, top=0.91)
    plt.plot(x, tcurve, color="red")
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.savefig("operator.png")
    plt.close()
    img_src = cv2.imread("img.png", 0)
    img_dst = cv2.LUT(img_src, tcurve)
    cv2.imwrite("img.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
    Histgram(img_src, "src")
    Histgram(img_dst, "dst")
    replace()


def spatial_flt(flt_ary):
    Table(flt_ary, "operator")
    for i in range(9):
        flt_ary[i//3][i%3] = eval(flt_ary[i//3][i%3])
    flt_ary = flt_ary.astype(np.float64)
    img_src = cv2.imread("img.png", 0)
    img_tmp = cv2.filter2D(img_src, -1, flt_ary)
    img_dst = cv2.convertScaleAbs(img_tmp)
    cv2.imwrite("img.png", img_dst)
    cnv_img.create_image(0, 0, image=convert(img_dst), anchor="nw")
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
    Spectrum(fshift, "src", True)
    Spectrum(ftmp, "dst", True)
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
    frame_buttons.configure(width=128, height=wid)
    canvas_src.configure(width=wid//2, height=wid//2)
    canvas_dst.configure(width=wid//2, height=wid//2)
    canvas_operator.configure(width=wid//2, height=wid//2)
    frame_parametor.configure(width=wid//2, height=wid//2)
    canvas_hist.configure(width=wid//2, height=wid//2)
    canvas_spec.configure(width=wid//2, height=wid//2)


def PixelValue(x=16, y=16):
    img_src = cv2.imread("god.png", 0)
    img_dst = cv2.imread("img.png", 0)
    src = ([[img_src[x-1][y-1], img_src[x][y-1], img_src[x+1][y-1]],
            [img_src[x-1][y  ], img_src[x][y  ], img_src[x+1][y  ]],
            [img_src[x-1][y+1], img_src[x][y+1], img_src[x+1][y+1]]])
    Table(src, "src")
    dst = ([[img_dst[x-1][y-1], img_dst[x][y-1], img_dst[x+1][y-1]],
            [img_dst[x-1][y  ], img_dst[x][y  ], img_dst[x+1][y  ]],
            [img_dst[x-1][y+1], img_dst[x][y+1], img_dst[x+1][y+1]]])
    Table(dst, "dst")


def Gaussian2D(width, sigma=4):
    x = np.linspace(-16, 16, width)
    y = np.linspace(-16, 16, width)
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
    global img_god, wid, size, center
    wid = min(img_god.shape[0], img_god.shape[1])
    img_god = img_god[0:wid, 0:wid]
    size = img_god.shape
    cv2.imwrite("god.png", img_god)
    cv2.imwrite("img.png", img_god)
    center = wid//2
    WidgetSize(wid)
    cnv_img.create_image(0, 0, image=convert(img_god), anchor="nw")


def shading(event):
    eval(cmb_shd.get())()


def filtering(event):
    eval(cmb_flt.get())()


def Values(event):
    frame_values.grid()
    img = cv2.imread("img.png", 0)
    Histgram(img, "hist")
    Spectrum(img, "spec")
    img_hist = cv2.imread("hist.png")
    img_spec = cv2.imread("spec.png")
    img_hist = cv2.resize(img_hist, (wid//2, wid//2))
    img_spec = cv2.resize(img_spec, (wid//2, wid//2))
    canvas_hist.create_image(0, 0, image=convert(img_hist), anchor="nw")
    canvas_spec.create_image(0, 0, image=convert(img_spec), anchor="nw")


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

btn_values = tk.Button(frame_buttons, text="Image Values")
btn_values.grid(row=5, column=0, sticky=tk.E+tk.W)
btn_values.bind("<1>", Values)

for i, child in enumerate(frame_buttons.winfo_children()):
    frame_buttons.grid_rowconfigure(i, weight=1)

frame_buttons.grid_columnconfigure(0, weight=1)

# Processing Detail
frame_process = tk.LabelFrame(root, text="Proccesing Detail")
frame_process.grid(row=0, column=2)

canvas_src = tk.Canvas(frame_process)
canvas_src.grid(row=0, column=0)

canvas_operator = tk.Canvas(frame_process)
canvas_operator.grid(row=0, column=1)

frame_parametor = tk.LabelFrame(frame_process, text="Proccesing Parameter")
frame_parametor.grid(row=1, column=0)
frame_parametor.grid_propagate(False)

lbl_R = tk.Label(frame_parametor, text="R")
lbl_R.grid(row=0, column=0)

scl_R = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=255)
scl_R.grid(row=0, column=1, sticky=tk.NSEW)
scl_R.set(75)

lbl_r = tk.Label(frame_parametor, text="r")
lbl_r.grid(row=1, column=0)

scl_r = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=255)
scl_r.grid(row=1, column=1, sticky=tk.NSEW)
scl_r.set(25)

lbl_sigma = tk.Label(frame_parametor, text="sigma")
lbl_sigma.grid(row=2, column=0)

scl_sigma = tk.Scale(frame_parametor, orient="horizontal", from_=1, to=8, resolution=0.2)
scl_sigma.grid(row=2, column=1, sticky=tk.NSEW)
scl_sigma.set(4)

lbl_Dsigma = tk.Label(frame_parametor, text="Dsigma")
lbl_Dsigma.grid(row=3, column=0)

scl_Dsigma = tk.Scale(frame_parametor, orient="horizontal", from_=1, to=8, resolution=0.2)
scl_Dsigma.grid(row=3, column=1, sticky=tk.NSEW)
scl_Dsigma.set(2)

lbl_rate = tk.Label(frame_parametor, text="rate")
lbl_rate.grid(row=4, column=0)

scl_rate = tk.Scale(frame_parametor, orient="horizontal", from_=1, to=2, resolution=0.2)
scl_rate.grid(row=4, column=1, sticky=tk.NSEW)
scl_rate.set(1.2)

frame_parametor.grid_columnconfigure(1, weight=1)

canvas_dst = tk.Canvas(frame_process)
canvas_dst.grid(row=1, column=1)

# Current Image Values
frame_values = tk.LabelFrame(root, text="Image Values")
frame_values.grid(row=0, column=3)

canvas_hist = tk.Canvas(frame_values)
canvas_hist.grid(row=0, column=0)

canvas_spec = tk.Canvas(frame_values)
canvas_spec.grid(row=1, column=0)

# --------------------------------------------------
# main
FileSelect()
cnv_img.create_image(0, 0, image=convert(img_god), anchor="nw")
DC = np.fft.fft2(img_god)
DC = np.fft.fftshift(DC)
DC = 20 * np.log(np.abs(DC))
DC_amp = np.max(DC)

root.mainloop()
