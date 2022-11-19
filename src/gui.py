import tkinter as tk
from tkinter import *
from tkinter import Text, filedialog, BOTH, W, E, N, S
from tkinter.ttk import Label, Style, Button, Frame
from PIL import Image, ImageTk
import cv2

root = Tk()
root.title("Face Recognition - Reigenface")

# Styles

s = Style()
s.configure('Mainframe.Tframe', background = '#d6f3ff')

# Images


# Frames

mainframe = Frame(root, width=1600, height=800, style='Mainframe.Tframe')
mainframe.grid(row=0, column=0, sticky='NSEW')

topframe = Frame(mainframe)
topframe.grid(row=0, column=0, sticky='NSEW', columnspan=2)

uploadphotoFrame = Frame(mainframe, width=300)
uploadphotoFrame.grid(row=1, column=0, padx=3, pady=3, sticky='NSEW')

conversionFrame = Frame(mainframe, width=1200)
conversionFrame.grid(row=1, column=1, padx=3, pady=3, sticky='NSEW')


# Conversion Frame section


# Execution
root.mainloop()