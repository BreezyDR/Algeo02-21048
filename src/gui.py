import tkinter as tk
from tkinter import *
from tkinter import Text, filedialog, BOTH, W, E, N, S
from tkinter.filedialog import askopenfile
from tkinter.ttk import Label, Style, Button, Frame
from PIL import Image, ImageTk
import cv2

import ctypes

from src.EigenSolver import EigenSolver
from src.file import getUniqueId, readFolder, readFile, readWebCam, writeArrayAsData, readDataAsArray

import time

def openImage(path):
    return PhotoImage(Image.open(path))
 
class GUIRunner():
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Face Recognition - Reigenface")
        self.root.geometry("1600x800")

        ctypes.windll.shcore.SetProcessDpiAwareness(1) 
        # it alters dpi
        # https://coderslegacy.com/python/problem-solving/improve-tkinter-resolution/

        # variables
        self.defaultImgDimension = 256

        bgcolor = '#e1e2e1'
        defaultImg = ImageTk.PhotoImage(Image.open('./public/default/default.jpg').resize((self.defaultImgDimension, self.defaultImgDimension), Image.ANTIALIAS))
        


        # Styles

        s = Style()
        
        s.configure('Mainframe.TFrame', background = bgcolor)
        s.configure('SubTitle.TLabel', font = ('Helvetica', 14), foreground = 'black', padding = (5, 5, 5, 5), width = 20, background = bgcolor)
        s.configure('Upload.TButton', font = ('arial', 10, 'underline'), foreground = 'blue', background= bgcolor)
        s.configure('Upload.TLabel', font = ('Helvetica', 10), foreground = 'blue', background= bgcolor)
        s.configure('Title.TLabel', font=('Helvetica', 20, 'bold'), foreground = 'black', padding = (5, 5, 5, 5), background=bgcolor)
        s.configure('Frameling.TFrame', background = bgcolor)


        # Frames
        self.root.grid_rowconfigure(0, weight=1)

        mainframe = Frame(self.root, width=1600, height=800, style='Mainframe.TFrame')
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

        uploadphotoFrame.grid_rowconfigure(6, weight=1)
        save_db_button = Button(uploadphotoFrame, text = 'Save Calculations', style='Upload.TButton', width=10,  command = lambda:self.save_db())
        save_db_button.grid(row=6, column=0, sticky='WE')

        uploadphotoFrame.grid_rowconfigure(7, weight=1)
        save_db_button = Button(uploadphotoFrame, text = 'Load Calculations', style='Upload.TButton', width=10,  command = lambda:self.load_db())
        save_db_button.grid(row=7, column=0, sticky='WE')
    
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
        resultLabel = Button(uploadphotoFrame, text='Get Result', style='Upload.TButton', command = lambda:self.solve_pca())
        resultLabel.grid(row=4, column=0, sticky='WE')

        uploadphotoFrame.grid_rowconfigure(5, weight=20)
        uploadphotoFrame.grid_columnconfigure(0, weight=2)
        uploadphotoFrame.grid_columnconfigure(1, weight=8)


        # Conversion Frame section

        testimageLabel = Label(conversionFrame, text='Test Image', style= 'SubTitle.TLabel')
        testimageLabel.grid(row=0, column=0, sticky='WE')

        self.test_panel = Label(conversionFrame)
        self.updateImage(self.test_panel, defaultImg)
        self.test_panel.grid(column=0,row=1, sticky='WE')

        
        

        closestResultLabel = Label(conversionFrame, text='Closest Result', style= 'SubTitle.TLabel')
        closestResultLabel.grid(row=0, column=1, sticky='WE')

        self.result_panel = Label(conversionFrame, image=None)
        self.updateImage(self.result_panel, defaultImg)
        self.result_panel.grid(row=1, column=1, sticky='WE')

        executionLabel = Label(conversionFrame, text='Execution time:', style='SubTitle.TLabel')
        executionLabel.grid(row=2, column=0, sticky='WE')


        self.web_cam_panel = Label(conversionFrame, image=None)
        self.updateImage(self.web_cam_panel, defaultImg)
        self.web_cam_panel.grid(row=2, column=1, sticky='we')

        self.webcam_text = ['Start Webcam Capture', 'Stop Webcam Capture']
        self.startWebCam = Button(conversionFrame, text=self.webcam_text[False], style='Upload.TButton', command = lambda:self.toggle_webcam())
        self.startWebCam.grid(row=2, column=2, sticky='WE')

        
        # object variables

        self.root = self.root
        self.style = s

        self.eigensolver = EigenSolver(self.defaultImgDimension)
        self.folder_path = None
        self.target_path = None

        self.result_image = None
        self.test_image = None

        self.webcam_active = False
        self.frame_capture_result = None

    def run(self):
        # Execution
        startTime = time.time()
        self.cap = cv2.VideoCapture(0)

        print('program started in', time.time() - startTime, 'seconds')
        
        self.root.mainloop()



     # Upload image
    def upload_trainfolder(self):
        folderName = filedialog.askdirectory(initialdir='./public')
       
        if folderName != '':
            files, files_path = readFolder(folderName)
            
            self.eigensolver.train(files=files, files_path=files_path)
            self.folder_path = folderName
        
        self.updateUI()

    def upload_targetImage(self):
        types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png'), ('Jpeg Files', '*.jpeg')]
        filename = filedialog.askopenfilename(multiple=False, filetypes=types, initialdir='./public')
        
        if filename != '':
            file, file_path = readFile(filename)
            self.eigensolver.solve(file, file_path)
            self.target_path = file_path[0]
            print(self.target_path)
            self.test_image = ImageTk.PhotoImage(Image.open(self.target_path).resize((self.defaultImgDimension, self.defaultImgDimension), Image.ANTIALIAS))
        
        self.updateUI()

    def load_db(self):
        types = [('Json Files', '*.json')]
        filename = filedialog.askopenfilename(multiple=False, initialdir='./public/data', filetypes=types)
        
        if filename != '':
            __img_cnt, __mean, __eigVec, __distributedWeight, __files_path = readDataAsArray(filename)

            # load dalam data
            self.eigensolver.trainImgCount = __img_cnt
            self.eigensolver.mean = __mean
            self.eigensolver.eigVec = __eigVec
            self.eigensolver.distributedWeight = __distributedWeight
            self.eigensolver.files_path = __files_path

            self.eigensolver.hasTrained = True
        

        print('finished loading ...')
        
        self.updateUI()


    def save_db(self):
        types = [('Json Files', '*.json')]
        filename = filedialog.asksaveasfilename(filetypes=types, initialdir='./public/data', initialfile='data_' + getUniqueId() + '.json')
        
        if filename != '' and self.eigensolver.hasTrained:
            writeArrayAsData(filename, self.eigensolver.trainImgCount, self.eigensolver.mean, self.eigensolver.eigVec, self.eigensolver.distributedWeight, self.eigensolver.files_path)

        print('finished saving ...')
        
        self.updateUI()

    def solve_pca(self):
        self.eigensolver.showResult()
        
        self.result_image = ImageTk.PhotoImage(Image.open(self.eigensolver.image_path).resize((self.defaultImgDimension, self.defaultImgDimension), Image.ANTIALIAS))

        self.updateUI()

        
    
    def updateImage(self, obj, img):
        obj.config(image = img)
        obj.photo = img
        

    def updateUI(self):
        if self.folder_path != None:
            self.path_label.configure(text=self.folder_path)
        
        if self.target_path != None:
            self.target_label.configure(text=str(self.target_path))

        if self.result_image != None:
            self.updateImage(self.result_panel, self.result_image)

        if self.test_image != None:
            self.updateImage(self.test_panel, self.test_image)

    def toggle_webcam(self):
        self.webcam_active = not self.webcam_active # switch state
        
        self.startWebCam.config(text=self.webcam_text[self.webcam_active])

        if self.webcam_active:
            self.get_webcam_data()
            

    def get_webcam_data(self):
        if not self.cap.isOpened:
            print('--(!)Error opening video capture')
            exit(0)

        
        ret, frame = self.cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            return

        self.frame_capture_result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        if self.webcam_active:
            self.updateImage(self.web_cam_panel, ImageTk.PhotoImage(image=Image.fromarray(self.crop_webcam(self.frame_capture_result))))
            self.root.after(1, self.get_webcam_data) # 0 will only make the program halted
        else:
            frame, path = readWebCam(cv2.cvtColor(self.crop_webcam(self.frame_capture_result), cv2.COLOR_RGB2GRAY))
            self.eigensolver.solve(frame, path)


    # crop webcam
    def crop_webcam(self, array):
        h, w, _ = array.shape
        start_w = (w - h) // 2
        
        return  cv2.resize(array[:, start_w:start_w+h], (self.defaultImgDimension, self.defaultImgDimension))