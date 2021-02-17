import dip
#
import numpy as np
import cv2
from PIL import Image, ImageTk
import pyocr
import pyocr.builders
import re

processes = ['Tesseract']

def Tesseract(image):
    tools = pyocr.get_available_tools()
    tool = tools[0]
    builder_line = pyocr.builders.LineBoxBuilder(tesseract_layout=6)
    result = tool.image_to_string(image.pil, lang='jpn', builder=builder_line)
    result_image = image.color
    for r in result:
        cv2.rectangle(result_image, r.position[0], r.position[1], (0, 0, 255), 2)
    dst = dip.Imeji(result_image)
    return dst
