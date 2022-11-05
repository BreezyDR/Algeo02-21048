import tkinter as tk
from tkinter import *
from tkinter import Text, filedialog, BOTH, W, E, N, S
from tkinter.ttk import Label, Style, Button, Frame
import cv2

class MainWeb(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.widget()
        self.place()
    def widget(self):
        self.master.title("Face Recognition Kelompok 14 Kelas 1")
        self.pack(fill=BOTH, expand=True)