import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from pickle import *
from helpers import *
from ImageProcessing import *
from PIL import Image
from scipy import ndimage, signal

# LOADING
root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files) # Loading percentage p of images
imgs = [load_image(image_dir + files[i]) for i in range(n)]

# DATA MANIPULATION

# Image equalization
for i in range(n):
    uint8 = img_float_to_uint8(imgs[i])
    r,g,b = cv2.split(uint8)
    uint8_equalized = cv2.merge((cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)))
    imgs[i] = (img_as_float(uint8_equalized));

# Edge Enhancement

for i, img in enumerate(imgs):
    imgs[i] = (ChannelAugmentation([img])[0][:,:,:3]
               
# SAVING DATA          

picklefile = open("X2DNoPatch3ChannelsNoAug"+".pickle",'wb')
pickle.dump(imgs,picklefile)
picklefile.close()

print("Saved X variable to "+"X2DNoPatch3ChannelsNoAug"+".pickle")


