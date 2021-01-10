

import cv2 as cv
from commonfunctions import *
#import cv 
import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import histogram 
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

from collections import Counter

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

matplotlib.rcParams['figure.dpi'] = 200


def getNotes(original_img,img_thresh):

    #ori_img = io.imread(pathImage)

    width= original_img.shape[1]
    height = original_img.shape[0]

    #img_thresh = ori_img
    show_images([img_thresh],["thresh"])

    hist = []

    # get Concentration Histogram
    for x in range(width):
        hist.append(sum(img_thresh[0:height,x] == 0))

    # find thr for detecting character
    occurence_count = Counter(hist)
    thr_character = occurence_count.most_common(1)[0][0]

    #convert list of hist to numpy array as uint8 for  using in countours
    a = np.zeros(img_thresh.shape)
    arr = np.array(hist)
    a = a < ((arr > thr_character)*255)
    a = a.astype('uint8')

    # using a that represnts hist list in countours
    contours, hierarchy = cv.findContours(a, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    listOfImages = []
   
    for contour in contours:

        x, y, w, h = cv.boundingRect(contour)
       
        out = original_img[y:y+h,x:x+w]
        listOfImages.append(out)
        
    return listOfImages    
