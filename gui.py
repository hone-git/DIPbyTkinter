# GUI関係
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
#
import dip
import ch4
import ch5

if __name__ == '__main__':
    first = dip.Image()
    first.fileopen()

    second = ch4.NegPosInversion(first)
    second.filesave()
