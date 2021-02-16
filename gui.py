# GUI関係
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
#
import dip
import ch4
import ch5
import ch6
import ocr

def selectChapter(event):
    chapter = chapter_box.get()
    process_box.configure(values=eval(chapter).processes)
    print('do')

def imageProcess(event):
    process = chapter_box.get() + '.' + process_box.get()
    print(process)
    second = eval(process)(garbage[-1])
    garbage.append(second)
    canvas.create_image(0, 0, image=second.tk, anchor='nw')


def imageOCR(event):
    print("ocr")
    _, second = ocr.Tesseract(garbage[-1])
    garbage.append(second)
    canvas.create_image(0, 0, image=second.tk, anchor='nw')


if __name__ == '__main__':
    chapters = ['ch4', 'ch5', 'ch6', 'ocr']

    garbage = []
    root = tk.Tk()
    root.title("dip")

    canvas = tk.Canvas(root)
    canvas.grid(row=0, column=0, sticky=tk.NSEW)
    canvas.bind('<1>', imageProcess)
    canvas.bind('<3>', imageOCR)

    chapter = tk.StringVar()

    chapter_box = ttk.Combobox(root, state="readonly", textvariable=chapter, values=chapters)
    chapter_box.grid(row=1, column=0, sticky=tk.NSEW)
    chapter_box.set(chapters[1])
    chapter_box.bind('<<ComboboxSelected>>', selectChapter)

    print(chapter)

    chapter.set(chapter_box.get())
    print(chapter_box.get())

    process_box = ttk.Combobox(root, state="readonly", values=eval(chapter).processes)
    process_box.grid(row=2, column=0, sticky=tk.NSEW)
    process_box.bind('<<ComboboxSelected>>', lambda e: print("A"))

    first = dip.Imeji()
    first.savefile()
    garbage.append(first)
    canvas.create_image(0, 0, image=first.tk, anchor='nw')
    canvas.configure(width=first.width, height=first.height)

    root.mainloop()
