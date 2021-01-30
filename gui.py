# GUI関係
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
#
import dip
import ch4
import ch5
import ch6

def click(event):
    second = ch5.Median(garbage[-1])
    garbage.append(second)
    canvas.create_image(0, 0, image=second.tk, anchor='nw')

if __name__ == '__main__':
    garbage = []
    root = tk.Tk()
    root.title("dip")

    canvas = tk.Canvas(root)
    canvas.grid(row=0, column=0, sticky=tk.NSEW)
    canvas.bind('<1>', click)

    first = dip.Imeji("tmp.png")
    first.savefile()
    garbage.append(first)
    canvas.create_image(0, 0, image=first.tk, anchor='nw')
    canvas.configure(width=first.width, height=first.height)

    root.mainloop()
