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
import itertools

imgs = []
x = np.arange(256)
old_x = None
old_y = None
mode = itertools.cycle(['Line', 'ProbabilisticLine', 'Circle'])
params = {'Line':["threshold"], 'ProbabilisticLine':["threshold", "minLineLength", "maxLineGap"],
          'Circle':["threshold", "minRadius", "maxRadius"]}

dir = expanduser("~") + "/Pictures/dipImage"
try:
    os.mkdir(dir)
except FileExistsError:
    pass
os.chdir(dir)


def convert(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    imgs.append(img_tk)
    return img_tk


def HoughLine(img_gray, img_bgr):
    img_gray = cv2.Canny(img_gray, 50, 150, apertureSize = 3)
    threshold = scl_threshold.get()
    lines = cv2.HoughLinesP(img_gray, rho=1, theta=np.pi/180, threshold=threshold)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img_bgr, (x1,y1), (x2,y2), (0,0,255), 3)
    return img_bgr


def HoughProbabilisticLine(img_gray, img_bgr):
    img_gray = cv2.Canny(img_gray, 50, 150, apertureSize = 3)
    threshold = scl_threshold.get()
    mLL = scl_minLineLength.get()
    mLG = scl_maxLineGap.get()
    lines = cv2.HoughLinesP(img_gray, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=mLL, maxLineGap=mLG)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img_bgr, (x1,y1), (x2,y2), (0,0,255), 3)
    return img_bgr


def HoughCircle(img_gray, img_bgr):
    img_gray = cv2.Canny(img_gray, 50, 150, apertureSize = 3)
    threshold = scl_threshold.get()
    minR = scl_minRadius.get()
    maxR = scl_maxRadius.get()
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=threshold, minRadius=minR, maxRadius=maxR)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img_bgr,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(img_bgr,(i[0],i[1]),2,(0,0,255),3)
    return img_bgr


def FileSelect():
    fTyp = [("", "*")]
    iDir = os.path.abspath(Path().resolve())
    file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    img_origin = cv2.imread(file, 0)
    wid_origin = img_origin.shape
    wid = max(img_origin.shape[0], img_origin.shape[1])
    img_square = np.full((wid, wid), 0)
    img_square[(wid-wid_origin[0])//2:(wid+wid_origin[0])//2, (wid-wid_origin[1])//2:(wid+wid_origin[1])//2] = img_origin
    cv2.imwrite("img.png", img_square)


def start(event):
    global old_x, old_y
    if old_x and old_y:
        cnv.create_line(old_x, old_y, event.x, event.y, width=3)
    old_x = event.x
    old_y = event.y


def stop(event):
    global old_x, old_y
    old_x = None
    old_y = None
    cnv.postscript(file="test.eps")
    im = Image.open("test.eps")
    im.save('test.png', "png")
    img_origin = cv2.imread("test.png")
    img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    for i, child in enumerate(frame_parametor.winfo_children()):
        child.grid_remove()
    button_mode.grid()
    for i in params[button_mode["text"]]:
        eval("lbl_"+i+".grid()")
        eval("scl_"+i+".grid()")
    img_hough = eval("Hough"+button_mode["text"])(img_gray, img_origin)
    img_resized = cv2.resize(img_hough, (256, 256))
    cnv_hough.create_image(0, 0, image=convert(img_resized), anchor="nw")


def ChangeMode(event):
    """
    """
    event.widget["text"] = next(mode)


def select(event):
    FileSelect()
    img = cv2.imread("img.png")
    img = cv2.resize(img, (512, 512))
    cnv.create_image(0, 0, image=convert(img), anchor="nw")

# --------------------------------------------------
# GUI

root = tk.Tk()
root.title("DigitalImageProcessing GUI")

cnv = tk.Canvas(root, bg="white", width=512, height=512)
cnv.grid(row=0, column=0)
cnv.bind('<B1-Motion>', start)
cnv.bind('<ButtonRelease-1>', stop)
cnv.bind('<3>', select)

frm = tk.LabelFrame(root, text="Hough Transform")
frm.grid(row=0, column=1)

cnv_hough = tk.Canvas(frm, width=256, height=256)
cnv_hough.grid(row=0, column=0)

frame_parametor = tk.LabelFrame(frm, text="Parameter", width=256, height=256)
frame_parametor.grid(row=1, column=0)
frame_parametor.grid_propagate(False)

lbl_threshold = tk.Label(frame_parametor, text="threshold")
lbl_threshold.grid(row=0, column=0)

scl_threshold = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=400)
scl_threshold.grid(row=0, column=1, sticky=tk.NSEW)
scl_threshold.set(40)

lbl_minLineLength = tk.Label(frame_parametor, text="minLineLength")
lbl_minLineLength.grid(row=1, column=0)

scl_minLineLength = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=800)
scl_minLineLength.grid(row=1, column=1, sticky=tk.NSEW)
scl_minLineLength.set(80)

lbl_maxLineGap = tk.Label(frame_parametor, text="maxLineGap")
lbl_maxLineGap.grid(row=2, column=0)

scl_maxLineGap = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=20)
scl_maxLineGap.grid(row=2, column=1, sticky=tk.NSEW)
scl_maxLineGap.set(5)

lbl_minRadius = tk.Label(frame_parametor, text="minRadius")
lbl_minRadius.grid(row=3, column=0)

scl_minRadius = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=200)
scl_minRadius.grid(row=3, column=1, sticky=tk.NSEW)
scl_minRadius.set(10)

lbl_maxRadius = tk.Label(frame_parametor, text="maxRadius")
lbl_maxRadius.grid(row=4, column=0)

scl_maxRadius = tk.Scale(frame_parametor, orient="horizontal", from_=0, to=400)
scl_maxRadius.grid(row=4, column=1, sticky=tk.NSEW)
scl_maxRadius.set(200)

button_mode = tk.Button(frame_parametor, text=next(mode))
button_mode.grid(row=5, column=0, columnspan=2, sticky=tk.NSEW)
button_mode.bind("<1>", ChangeMode)

for i, child in enumerate(frame_parametor.winfo_children()):
    frame_parametor.grid_rowconfigure(i, weight=1)
    child.grid_remove()

frame_parametor.grid_columnconfigure(1, weight=1)


root.mainloop()
