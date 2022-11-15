import tkinter as tk
from tkinter import *
from tkinter import Text, filedialog, BOTH, W, E, N, S
from tkinter.ttk import Label, Style, Button, Frame
import cv2

class MainGUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.widget()
        self.place()
    def widget(self):
        self.pack(fill=BOTH, expand=True)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=4)
        


run = tk.Tk()
run.title("Face Recognition")
run.geometry("400x300 + 300x300")
gui = MainGUI(master=run)
gui.mainloop()