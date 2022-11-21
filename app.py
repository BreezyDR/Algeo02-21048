import cv2
import numpy as np

from src.file import readFolder, readFile

from src.EigenSolver import EigenSolver
from src.gui import GUIRunner

if __name__ == '__main__':

    # initialize gui
    gui = GUIRunner()

    # # read files streams
    # files_path = 'public/images/'
    # target_path = 'public/target'
    # files, individual_files_path = readFolder(files_path)
    # new_files, individual_target_path = readFolder(target_path)

    # # desired image size
    # size = 256

    # eigenSolver = EigenSolver(desiredSize=size)

    # eigenSolver.train(files=files, files_path=individual_files_path)
    # eigenSolver.solve(new_files=new_files, new_files_path=individual_target_path)
    
    # eigenSolver.showResult()
    

    gui.run()
    