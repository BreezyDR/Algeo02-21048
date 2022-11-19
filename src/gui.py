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
s.configure('SubTitle.TLabel', font = ('Helvetica', 14, 'bold'), foreground = 'black', padding = (5, 5, 5, 5), width = 20)
s.configure('Upload.TButton', font = ('arial', 10, 'underline'), foreground = 'blue')

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


# Upload Photo & Database section
uploaddatatestLabel = Label(uploadphotoFrame, text="Masukkan data test Anda", style= 'SubTitle.TLabel')
uploaddatatestLabel.grid(row=0, column=0, sticky= 'WE')
uploaddatatestLabel.configure(anchor='center')

datatestButton = Button(uploadphotoFrame, text = 'Choose File', style='Upload.TButton')
datatestButton.grid(row=1, column=0, sticky='WE')

uploadtestphotoLabel = Label(uploadphotoFrame, text="Masukkan image Anda", style= 'SubTitle.TLabel')
uploadtestphotoLabel.grid(row=2, column=0, sticky= 'WE')
uploadtestphotoLabel.configure(anchor='center')

testphotoButton = Button(uploadphotoFrame, text = 'Choose File', style='Upload.TButton')
testphotoButton.grid(row=3, column=0, sticky='WE')


# Conversion Frame section


# Execution
root.mainloop()