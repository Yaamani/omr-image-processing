#!/usr/bin/env python
# coding: utf-8

# In[1]:


from getnotes import *
from line_extraction import *
import cv2 as cv


# In[2]:


def segment_into_lines_of_notes(img):
    gray_lines , binary_lines  = get_lines(img)
    my_show_images(binary_lines)
    lines_of_notes = []
    for i in range(len(gray_lines)):
    
        lines_of_notes.append(getNotes(gray_lines[i] ,binary_lines[i]))
    
    return lines_of_notes

