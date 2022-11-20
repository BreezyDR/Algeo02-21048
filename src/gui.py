import tkinter as tk
from tkinter import *
from tkinter import Text, filedialog, BOTH, W, E, N, S
from tkinter.filedialog import askopenfile
from tkinter.ttk import Label, Style, Button, Frame
from PIL import Image, ImageTk
import cv2

import ctypes
 


root = Tk()
root.title("Face Recognition - Reigenface")
root.geometry("1600x800")

ctypes.windll.shcore.SetProcessDpiAwareness(1) 
# it alters dpi
# https://coderslegacy.com/python/problem-solving/improve-tkinter-resolution/

# Styles

s = Style()
s.configure('Mainframe.TFrame', background = '#d6f3ff')
s.configure('SubTitle.TLabel', font = ('Helvetica', 14), foreground = 'black', padding = (5, 5, 5, 5), width = 20)
s.configure('Upload.TButton', font = ('arial', 10, 'underline'), foreground = 'blue')
s.configure('Title.TLabel', font=('Helvetica', 20, 'bold'), foreground = 'black', padding = (5, 5, 5, 5))


# Upload image
def upload_image():
    types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png'), ('Jpeg Files', '*.jpeg')]
    filename = filedialog.askopenfilename(multiple=False, filetypes=types)
    testphoto = Image.open(filename)
    testphoto = testphoto.resize((400,400), Image.ANTIALIAS)
    testphoto = ImageTk.PhotoImage(testphoto)
    e1 = Label(conversionFrame, image=testphoto)
    e1.grid(row=1,column=0,sticky='WE')


# Frames

mainframe = Frame(root, width=1600, height=800, style='Mainframe.TFrame')
mainframe.grid(row=0, column=0, sticky='NSEW')

topframe = Frame(mainframe)
topframe.grid(row=0, column=0, sticky='NSEW', columnspan=2)

uploadphotoFrame = Frame(mainframe, width=300)
uploadphotoFrame.grid(row=1, column=0, padx=3, pady=3, sticky='NSEW')

conversionFrame = Frame(mainframe, width=1200)
conversionFrame.grid(row=1, column=1, padx=3, pady=3, sticky='NSEW')

# Top Frame Section
titleLabel = Label(topframe, text="Face Recognition - Reigenface", style='Title.TLabel')
titleLabel.place(relx=0.5,rely=0.5,anchor=CENTER)

# Upload Photo & Database section
uploaddatatestLabel = Label(uploadphotoFrame, text="Masukkan data test Anda", style= 'SubTitle.TLabel')
uploaddatatestLabel.grid(row=0, column=0, sticky= 'WE')

datatestButton = Button(uploadphotoFrame, text = 'Choose File', style='Upload.TButton',width=10)
datatestButton.grid(row=1, column=0, sticky='WE')

uploadtestphotoLabel = Label(uploadphotoFrame, text="Masukkan image Anda", style= 'SubTitle.TLabel')
uploadtestphotoLabel.grid(row=2, column=0, sticky= 'WE')

testphotoButton = Button(uploadphotoFrame, text = 'Choose File', style='Upload.TButton',width=10,command=lambda:upload_image())
testphotoButton.grid(row=3, column=0, sticky='WE')

resultLabel = Label(uploadphotoFrame, text='Result', style='Upload.TButton')
resultLabel.grid(row=4, column=0, sticky='WE')


# Conversion Frame section

testimageLabel = Label(conversionFrame, text='Test Image', style= 'SubTitle.TLabel')
testimageLabel.grid(row=0, column=0, sticky='WE')

closestResultLabel = Label(conversionFrame, text='Closest Result', style= 'SubTitle.TLabel')
closestResultLabel.grid(row=0, column=1, sticky='WE')

executionLabel = Label(conversionFrame, text='Execution time:', style='SubTitle.TLabel')
executionLabel.grid(row=2, column=0, sticky='WE')


# Execution
root.mainloop()