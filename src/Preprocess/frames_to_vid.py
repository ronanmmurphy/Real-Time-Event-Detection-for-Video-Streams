import cv2
import numpy as np
import os
from os.path import isfile, join
import glob
import re


#file joins all the frames in folder and outputs avi video file

pathIn= './out_frames'
pathOut = 'output.avi'
fps = 15
frame_array = []

def keyfunc(name):
    nondigits = re.compile("\D")
    return int(nondigits.sub("",name))
    
    
#iterate through images in folder and sort by key
images = glob.glob("out_frames/*.jpg")
files=[]
for image in sorted(images, key=keyfunc):
    files.append(image)


#write each frame to video using opencv
for i in range(len(files)):
    filename=files[i]
    print(filename)
    #reading each files
    img = cv2.imread(filename)
    
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()