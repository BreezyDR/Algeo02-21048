import tkinter as tk
from tkinter import *
from tkinter import Text, filedialog, BOTH, W, E, N, S
from tkinter.filedialog import askopenfile
from tkinter.ttk import Label, Style, Button, Frame
from PIL import Image, ImageTk
import cv2

import ctypes

from src.EigenSolver import EigenSolver
from src.file import readFolder, readFile

def openImage(path):
    return PhotoImage(Image.open(path))
 
class GUIRunner():
    def __init__(self) -> None:
        root = Tk()
        root.title("Face Recognition - Reigenface")
        root.geometry("1600x800")

        ctypes.windll.shcore.SetProcessDpiAwareness(1) 
        # it alters dpi
        # https://coderslegacy.com/python/problem-solving/improve-tkinter-resolution/

        # Styles

        s = Style()
        bgcolor = '#e1e2e1'
        s.configure('Mainframe.TFrame', background = bgcolor)
        s.configure('SubTitle.TLabel', font = ('Helvetica', 14), foreground = 'black', padding = (5, 5, 5, 5), width = 20, background = bgcolor)
        s.configure('Upload.TButton', font = ('arial', 10, 'underline'), foreground = 'blue', background= bgcolor)
        s.configure('Upload.TLabel', font = ('Helvetica', 10), foreground = 'blue', background= bgcolor)
        s.configure('Title.TLabel', font=('Helvetica', 20, 'bold'), foreground = 'black', padding = (5, 5, 5, 5), background=bgcolor)
        s.configure('Frameling.TFrame', background = bgcolor)
        

        # Frames
        root.grid_rowconfigure(0, weight=1)

        mainframe = Frame(root, width=1600, height=800, style='Mainframe.TFrame')
        mainframe.grid_rowconfigure(0, weight=1)
        mainframe.grid_rowconfigure(1, weight=7)
        mainframe.columnconfigure(0, weight= 2)
        mainframe.columnconfigure(1, weight=1)
        
        mainframe.pack(expand=True, fill=BOTH)

        topframe = Frame(mainframe, style='Frameling.TFrame')
        topframe.grid(row=0, column=0, sticky='nsew')

        uploadphotoFrame = Frame(mainframe, width=300, style='Frameling.TFrame')
        # uploadphotoFrame.grid_rowconfigure(0, weight=1, )
        uploadphotoFrame.grid(row=1, column=0, padx=3, pady=3, sticky='nsew')

        conversionFrame = Frame(mainframe, width=1200, style='Frameling.TFrame')
        conversionFrame.grid(row=1, column=1, sticky='nsew')

        # Top Frame Section
        titleLabel = Label(topframe, text="Face Recognition - Reigenface", style='Title.TLabel')
        titleLabel.place(relx=0.5,rely=0.5,anchor=CENTER)
        titleLabel.grid_columnconfigure(0, weight=1)

        # Upload Photo & Database section
        uploadphotoFrame.grid_rowconfigure(0, weight=3)
        uploaddatatestLabel = Label(uploadphotoFrame, text="Masukkan data test Anda", style= 'SubTitle.TLabel')
        uploaddatatestLabel.grid(row=0, column=0, sticky= 'WE')

        uploadphotoFrame.grid_rowconfigure(1, weight=2)
        datatestButton = Button(uploadphotoFrame, text = 'Choose Folder', style='Upload.TButton', width=10,  command = lambda:self.upload_trainfolder())
        datatestButton.grid(row=1, column=0, sticky='WE')
    
        self.path_label = Label(uploadphotoFrame, text = 'Belum memasukkan folder', style='Upload.TLabel', width=20)
        self.path_label.grid(row=1, column=1, sticky='WE')

        uploadphotoFrame.grid_rowconfigure(2, weight=3)
        uploadtestphotoLabel = Label(uploadphotoFrame, text="Masukkan image Anda", style= 'SubTitle.TLabel')
        uploadtestphotoLabel.grid(row=2, column=0, sticky= 'WE')

        uploadphotoFrame.grid_rowconfigure(3, weight=2)
        testphotoButton = Button(uploadphotoFrame, text = 'Choose File', style='Upload.TButton',width=10,command = lambda:self.upload_targetImage())
        testphotoButton.grid(row=3, column=0, sticky='WE')

        self.target_label = Label(uploadphotoFrame, text = 'Belum memasukkan file', style='Upload.TLabel', width=20)
        self.target_label.grid(row=3, column=1, sticky='WE')

        uploadphotoFrame.grid_rowconfigure(4, weight=2)
        resultLabel = Button(uploadphotoFrame, text='Result', style='Upload.TButton', command = lambda:self.solve_pca())
        resultLabel.grid(row=4, column=0, sticky='WE')

        uploadphotoFrame.grid_rowconfigure(5, weight=20)
        uploadphotoFrame.grid_columnconfigure(0, weight=2)
        uploadphotoFrame.grid_columnconfigure(1, weight=8)


        # Conversion Frame section

        testimageLabel = Label(conversionFrame, text='Test Image', style= 'SubTitle.TLabel')
        testimageLabel.grid(row=0, column=0, sticky='WE')

        closestResultLabel = Label(conversionFrame, text='Closest Result', style= 'SubTitle.TLabel')
        closestResultLabel.grid(row=0, column=1, sticky='WE')

        executionLabel = Label(conversionFrame, text='Execution time:', style='SubTitle.TLabel')
        executionLabel.grid(row=2, column=0, sticky='WE')

        e1 = Label(conversionFrame, image=None)
        e1.grid(row=1,column=0,sticky='WE')

        self.root = root
        self.style = s

        self.eigensolver = EigenSolver(256)
        self.folder_path = None
        self.target_path = None
    

    def run(self):
        # Execution
        self.root.mainloop()



     # Upload image
    def upload_targetImage(self):
        types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png'), ('Jpeg Files', '*.jpeg')]
        filename = filedialog.askopenfilename(multiple=False, filetypes=types)
        # testphoto = Image.open(filename)
        # testphoto = testphoto.resize((400,400), Image.ANTIALIAS)
        # testphoto = ImageTk.PhotoImage(testphoto)
        # e1 = Label(conversionFrame, image=testphoto)
        # e1.grid(row=1,column=0,sticky='WE')
        if filename != '':
            # self.targetFiles = [filename] # we forced AN image path as folder paths
            file, file_path = readFile(filename)
            self.eigensolver.solve(file, file_path)
            self.target_path = file_path
        
        self.updateUI()

    def upload_trainfolder(self):
        folderName = filedialog.askdirectory()
        # testphoto = Image.open(filename)
        # testphoto = testphoto.resize((400,400), Image.ANTIALIAS)
        # testphoto = ImageTk.PhotoImage(testphoto)
        # e1 = Label(conversionFrame, image=testphoto)
        # e1.grid(row=1,column=0,sticky='WE')
        if folderName != '':
            files, files_path = readFolder(folderName)
            # print(self.folder_path, 'dsaf')
            self.eigensolver.train(files=files, files_path=files_path)
            self.folder_path = folderName
        
        self.updateUI()

    def solve_pca(self):
        self.eigensolver.showResult()
        print(self.eigensolver.new_files_path)

        # self.e1.configure(image=openImage(self.eigensolver.new_files_path[0]))

    def updateUI(self):
        self.path_label.configure(text=self.folder_path)
        self.target_label.configure(text=str(self.target_path))
