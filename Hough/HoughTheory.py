"""
"""
# ファイル関係
import os
from os.path import expanduser
from pathlib import Path
# 画像処理関係
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image, ImageTk
# GUI関係
import tkinter as tk
import tkinter.filedialog

plt.rcParams["figure.subplot.left"] = 0
plt.rcParams["figure.subplot.bottom"] = 0
plt.rcParams["figure.subplot.right"] = 1
plt.rcParams["figure.subplot.top"] = 1

dir = expanduser("~") + "/Pictures/dipImage"
try:
    os.mkdir(dir)
except FileExistsError:
    pass
os.chdir(dir)

wid = 512

imgs = []
X = np.linspace(0, 1, wid)
Y = np.linspace(1, 0, wid)
XY = []
THETE = np.linspace(0, np.pi, wid)
RHO = np.linspace(np.sqrt(2), -np.sqrt(2), wid)
TR = []

fig_TR, ax_TR = plt.subplots()
ax_TR.set_xlim(0, np.pi)
ax_TR.set_ylim(-np.sqrt(2), np.sqrt(2))
fig_XY, ax_XY = plt.subplots()
ax_XY.set_xlim(0, 1)
ax_XY.set_ylim(0, 1)

# --------------------------------------------------
# 関数


def convert(img_bgr):
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        img_rgb = img_bgr
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    imgs.append(img_tk)
    return img_tk


def plotTheteRho(x, y):
    ax_TR.plot(THETE, x*np.cos(THETE)+y*np.sin(THETE), color='black', linewidth=0.1)
    fig_TR.savefig("TheteRho.png")
    img = cv2.imread("TheteRho.png")
    img = cv2.resize(img, (512, 512))
    cnv_TR.create_image(0, 0, image=convert(img), anchor="nw")
    for i, j in TR:
        cnv_TR.create_oval(i-2, j-2, i+2, j+2, fill="black")


def plotXY(thete, rho):
    ax_XY.plot(X, -X/np.tan(thete)+rho/np.sin(thete), color='red')
    fig_XY.savefig("XY.png")
    img = cv2.imread("XY.png")
    img = cv2.resize(img, (512, 512))
    cnv_XY.create_image(0, 0, image=convert(img), anchor="nw")
    for i, j in XY:
        cnv_XY.create_oval(i-2, j-2, i+2, j+2, fill="black")


def updateXY(event):
    strX.set("x = "+str(round(X[event.x], 2)))
    strY.set("y = "+str(round(Y[event.y], 2)))


def selectXY(event):
    x = event.x
    y = event.y
    XY.append((x, y))
    cnv_XY.create_oval(x-2, y-2, x+2, y+2, fill="black")
    x = round(X[x], 2)
    y = round(Y[y], 2)
    lbl_x_slct["text"] = "Selected x = " + str(x)
    lbl_y_slct["text"] = "Selected y = " + str(y)
    lbl_funcTR["text"] = "ρ = " + str(x) + "*cos(θ) + " + str(y) + "*sin(θ)"
    plotTheteRho(x, y)


def updateTheteRho(event):
    strThete.set("thete = "+str(round(THETE[event.x], 2)))
    strRho.set("rho = "+str(round(RHO[event.y], 2)))


def selectTheteRho(event):
    x = event.x
    y = event.y
    TR.append((x, y))
    cnv_TR.create_oval(x-2, y-2, x+2, y+2, fill="black")
    thete = round(THETE[x], 2)
    rho= round(RHO[y], 2)
    lbl_thete_slct["text"] = "Selected θ = " + str(thete)
    lbl_rho_slct["text"] = "Selected ρ = " + str(rho)
    lbl_funcXY["text"] = "y = -x/tan(" + str(thete) + ") + " + str(rho) + "/sin(" + str(thete) + ")"
    plotXY(thete, rho)


def select(event):
    file = tkinter.filedialog.askopenfilename()
    img_origin = cv2.imread(file, 0)
    wid_origin = img_origin.shape
    wid = max(img_origin.shape[0], img_origin.shape[1])
    img_square = np.full((wid, wid), 0)
    img_square[(wid-wid_origin[0])//2:(wid+wid_origin[0])//2, (wid-wid_origin[1])//2:(wid+wid_origin[1])//2] = img_origin
    cv2.imwrite("img.png", img_square)
    img = cv2.imread("img.png", 0)
    img = cv2.Canny(img, 100, 200)
    img = cv2.resize(img, (512, 512))
    cnv_XY.create_image(0, 0, image=convert(img), anchor="nw")


# --------------------------------------------------
# GUI

root = tk.Tk()
root.title("DigitalImageProcessing GUI")

cnv_XY = tk.Canvas(root, bg="white", width=wid, height=wid)
cnv_XY.grid(row=0, column=0)
cnv_XY.bind('<Motion>', updateXY)
cnv_XY.bind('<1>', selectXY)
cnv_XY.bind('<3>', select)

frm = tk.Frame(root)
frm.grid(row=0, column=1)

frm_XY = tk.LabelFrame(frm, text="Select x-y", width=wid//2, height=wid//8)
frm_XY.grid(row=0, column=0)
frm_XY.grid_propagate(False)

strX = tk.StringVar(value="x = 0")
strY = tk.StringVar(value="y = 0")

lbl_x = tk.Label(frm_XY, textvariable=strX)
lbl_x.grid(row=0, column=0)

lbl_y = tk.Label(frm_XY, textvariable=strY)
lbl_y.grid(row=1, column=0)

for i, child in enumerate(frm_XY.winfo_children()):
    frm_XY.grid_rowconfigure(i, weight=1)
    child.configure(font=("CambliaMath", 12))

frm_XY.grid_columnconfigure(0, weight=1)

frm_XY2TR = tk.LabelFrame(frm, text="x-y to thete-rho", width=wid//2, height=wid//8*3)
frm_XY2TR.grid(row=1, column=0)
frm_XY2TR.grid_propagate(False)

lbl_x_slct = tk.Label(frm_XY2TR, text="Selected x is ...")
lbl_x_slct.grid(row=0, column=0)

lbl_y_slct = tk.Label(frm_XY2TR, text="Selected y is ...")
lbl_y_slct.grid(row=1, column=0)

lbl_funcTR = tk.Label(frm_XY2TR, text="ρ = x*cos(θ) + y*sin(θ)")
lbl_funcTR.grid(row=2, column=0)

for i, child in enumerate(frm_XY2TR.winfo_children()):
    frm_XY2TR.grid_rowconfigure(i, weight=1)
    child.configure(font=("CambliaMath", 12))

frm_XY2TR.grid_columnconfigure(0, weight=1)

frm_TR = tk.LabelFrame(frm, text="Selected thete-rho", width=wid//2, height=wid//8)
frm_TR.grid(row=2, column=0)
frm_TR.grid_propagate(False)

strThete = tk.StringVar(value="θ = 0")
strRho = tk.StringVar(value="ρ = 0")

lbl_thete = tk.Label(frm_TR, textvariable=strThete)
lbl_thete.grid(row=0, column=0)

lbl_rho = tk.Label(frm_TR, textvariable=strRho)
lbl_rho.grid(row=1, column=0)

for i, child in enumerate(frm_TR.winfo_children()):
    frm_TR.grid_rowconfigure(i, weight=1)
    child.configure(font=("CambliaMath", 12))

frm_TR.grid_columnconfigure(0, weight=1)

frm_TR2XY = tk.LabelFrame(frm, text="x-y to thete-rho", width=wid//2, height=wid//8*3)
frm_TR2XY.grid(row=3, column=0)
frm_TR2XY.grid_propagate(False)

lbl_thete_slct = tk.Label(frm_TR2XY, text="Selected θ is ...")
lbl_thete_slct.grid(row=0, column=0)

lbl_rho_slct = tk.Label(frm_TR2XY, text="Selected ρ is ...")
lbl_rho_slct.grid(row=1, column=0)

lbl_funcXY = tk.Label(frm_TR2XY, text="y = -x/tan(θ) + ρ/sin(θ)")
lbl_funcXY.grid(row=2, column=0)

for i, child in enumerate(frm_TR2XY.winfo_children()):
    frm_TR2XY.grid_rowconfigure(i, weight=1)
    child.configure(font=("CambliaMath", 12))

frm_TR2XY.grid_columnconfigure(0, weight=1)

cnv_TR = tk.Canvas(root, width=wid, height=wid)
cnv_TR.grid(row=0, column=2)
cnv_TR.bind('<Motion>', updateTheteRho)
cnv_TR.bind('<1>', selectTheteRho)

root.mainloop()
