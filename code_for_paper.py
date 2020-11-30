n#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:23:28 2020

@author: sanjana
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage.color import rgb2gray
from skimage.io import imread, imshow
from scipy import ndimage
from PIL import Image,ImageChops
from colours import color_analysis1
import math 
import os
import glob 
import random

df=pd.read_csv('/Users/sanjana/Desktop/IP Paper/Responses Modified - Form Responses 1.csv')
df.drop('Timestamp', axis=1, inplace=True)
df.drop('Kindly upload the image here ', axis=1, inplace=True)
print(df.head())
list(df.columns)
df.shape

img_dir = "/Users/sanjana/Desktop/IP Paper/tongueimgs" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 

for i in range(2,91):
    for f1 in files:
        tmp=f1.split("/")
        n=tmp[-1].split(".")    
        if(i==int(n[0])):
            img = cv2.imread(f1) 
            img=cv2.resize(img, (560, 560))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            data.append(img)
            break

df['pic']=data
print(df.head())

#Segmentation 2,0
img=data[0]
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (1,1,545,545) #rect = (start_x, start_y, width, height)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
fimg = img*mask2[:,:,np.newaxis]


fig=plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(121)
ax1.imshow(img)

ax2 = fig.add_subplot(122)
ax2.imshow(fimg)

#plt.colorbar()
plt.show()

fimg=cv2.cvtColor(fimg,cv2.COLOR_BGR2RGB)
cv2.imwrite('/Users/sanjana/Desktop/IP Paper/segmented/s2.jpg', fimg) 


img_dir = "/Users/sanjana/Desktop/IP Paper/segmented" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 

for i in range(2,91):
    for f1 in files:
        tmp=f1.split("/")
        a=tmp[-1].split("s")
        b=a[1].split(".")
        n=b[0]
        if(i==int(n)):
            img = cv2.imread(f1) 
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            data.append(img)
            break
        
df['segmented_img']=data
print(df1.head())

img=df1['segmented_img'][0]
plt.imshow(img)
plt.imshow(img,cmap="gray")

#Filter
kernel = np.array([[ 1, 1, 1,1,1],
                        [ 1, 1, 1,1,1],
                        [ 1, 1, 1,1,1],
                        [1, 1, 1,1,1], 
                        [1, 1, 1,1,1]
                       ])

print(df.head())
#Area
area=[]
for img in df['segmented_img']:
    cnt=0
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.filter2D(img, -1, kernel)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[0]):
            if(img[i][j]!=0):
                cnt+=1

    area.append(cnt)

df['area']=area
print(df.head())

'''#Area
area=[]
triangle_area=[]
triangle_area_ratio=[]
for img in df['segmented_img']:
    img=df['segmented_img'][0]
    cnt=0
    b=0
    h=0
    m=0
    tri_a=0
    points=[]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.filter2D(img, -1, kernel)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[0]):
            if(img[i][j]!=0):
                cnt+=1
                tup=(i,j)
                points.append(tup)
                
    y_max = max(points)[0]
    y_min= min(points)[0]
    x_max=max(points)[1]
    b=y_max-y_min
    m=(y_max+y_min)/2
    print(y_max,y_min,x_max,b,m)
    op1 = [it for it in points if it[0] == y_max]
    ind=int(len(op1)/2)
    p1=op1[ind]
    op2 = [it1 for it1 in points if it1[0] == y_min]
    ind2=int(len(op2)/2)
    p2=op2[ind2]
    
    x_m=min(op)[0]
    h=x_max-x_m
    tri_a=0.5*b*h
    tri_a_rat=tri_a/cnt
    
    cv2.line(img, p1, p2, (0,0,0), 2)
    plt.imshow(img,cmap="gray")


    area.append(cnt)
    triangle_area.append(tri_a)
    triangle_area_ratio.append(tri_a_rat)
    
    
df['area']=area
df['triangle_area']=triangle_area
df['triangle_area_ratio']=triangle_area_ratio
print(df.head())'''

#width & height & ratio
central_width=[]
central_height=[]
height_width_ratio=[]
mid_x=280
mid_y=280
for img in df['segmented_img']:
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img=cv2.filter2D(img, -1, kernel)
    left=0
    right=0
    top=0
    bot=0
    for i in range(mid_x,0,-1):
        if(img[mid_x][i]!=0):
            left+=1
            
    for j in range(mid_x,560):
        if(img[mid_x][j]!=0):
            right+=1
    
    for k in range(mid_y,0,-1):
        if(img[k][mid_y]!=0):
            top+=1
            
    for l in range(mid_y,560):
        if(img[l][mid_y]!=0):
            bot+=1
    
    h=top+bot           
    w=left+right
    h_w=h/w
    height_width_ratio.append(h_w)
    central_width.append(w)
    central_height.append(h)
    
df['height_width_ratio']=height_width_ratio
df['central_width']=central_width
df['central_height']=central_height
print(df.head())

#Smaller-Half-distance, Circle Area, Square Area
smaller_half_dist=[]
circle_area=[]
circle_area_ratio=[]
square_area=[]
square_area_ratio=[]
small=0
circ_a=0

for a in df.index: 
    small=min(df['central_width'][a],df['central_height'][a])
    shd=small/2
    circ_a=math.pi*(shd**2)
    circ_a_rat=circ_a/df['area'][a]
    square_a=4*(shd**2)
    square_a_rat=square_a/df['area'][a]
    square_area.append(square_a)
    square_area_ratio.append(square_a_rat)
    smaller_half_dist.append(shd)
    circle_area.append(circ_a)
    circle_area_ratio.append(circ_a_rat)
    

df['smaller_half_dist']=smaller_half_dist
df['circle_area']=circle_area
df['circle_area_ratio']=circle_area_ratio
print(df.head())


df.to_csv (r'/Users/sanjana/Desktop/IP Paper/dataframe_mod.csv', index = False, header=True)
df=pd.read_csv('/Users/sanjana/Desktop/IP Paper/dataframe_mod.csv')
print(df.head())

#Colour
from collections import Counter
import webcolors
import cv2
from PIL import Image 

COLOURS = {"red": (255, 0, 0),
              "green" : (0,255,0),
              "blue":(0,0,255),
              "pink":(255,192,203),
              "white":(255,250,250),
              "yellow":(255,255,0),
              "purple":(221,160,221)
              }

def classify(rgb_tuple):
    
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb_tuple) for k, v in COLOURS.items()}
    color = min(distances, key=distances.get)
    return color


def color_analysis(image):

   # initialize a Counter starting from zero
   color_counter = Counter({color: 0 for color in Counter(COLOURS)})
   for pixel_count, RGB in image.getcolors(image.width * image.height):
       if(RGB!=(0,0,0)):
           color_name = classify(RGB)
           color_counter[color_name] += pixel_count

   # Calculate percent for each color
   for color in color_counter:
       pixel_count = image.width * image.height
       color_counter[color] = color_counter[color] / pixel_count

   #del color_counter['black'] 
   colour = max(color_counter, key=color_counter.get)

   return colour

f1='/Users/sanjana/Desktop/IP Paper/segmented/s2.jpg'
img1 = Image.open(f1).convert('RGB')
col1=color_analysis1(f1)

f2='/Users/sanjana/Desktop/IP Paper/segmented/s81.jpg'
img2 = Image.open(f2).convert('RGB')
col2=color_analysis1(f2)

f3='/Users/sanjana/Desktop/IP Paper/segmented/s70.jpg'
img3 = Image.open(f3).convert('RGB')
col3=color_analysis1(f3)

f4='/Users/sanjana/Desktop/IP Paper/segmented/s21.jpg'
img4 = Image.open(f4).convert('RGB')
col4=color_analysis1(f4)


fig=plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(141)
ax1.set_title(col1)
ax1.imshow(img1)

ax2 = fig.add_subplot(142)
ax2.set_title(col2)
ax2.imshow(img2)

ax3 = fig.add_subplot(143)
ax3.set_title(col3)
ax3.imshow(img3)

ax4 = fig.add_subplot(144)
ax4.set_title(col4)
ax4.imshow(img4)

#plt.colorbar()
plt.show()

img_dir = "/Users/sanjana/Desktop/IP Paper/segmented" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
  
colour=[]
for i in range(2,91):
    for f1 in files:
        tmp=f1.split("/")
        a=tmp[-1].split("s")
        b=a[1].split(".")
        n=b[0]
        if(i==int(n)):
            img = Image.open(f1).convert('RGB')
            col=color_analysis(img)
            colour.append(col)
            
df['colour']=colour
print(df.head())


df.colour.unique() 

'''COLOURS = {"red": (255, 0, 0),
              "green" : (0,255,0),
              "blue":(0,0,255),
              "pink":(255,182,193),
              "white":(245,245,245),
              "yellow":(255,255,0),
              "purple":(218,112,214)
              }

img = Image.open(r"/Users/sanjana/Desktop/IP Paper/segmented/s21.jpg").convert('RGB')
col=color_analysis(img)
print(col)'''

#Contours (Texture/Patches)
num_cont=[]
area_cont=[]
per_cont=[]
for i in range(2,91):
    for f1 in files:
        tmp=f1.split("/")
        a=tmp[-1].split("s")
        b=a[1].split(".")
        n=b[0]
        if(i==int(n)):
            img = cv2.imread(f1,1)
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3),0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            filtered = []
            tot_are=0
            tot_per=0
            for c in contours:
                if cv2.contourArea(c) < 1000:
                    continue
                filtered.append(c)
                
            for c in filtered:
                area = cv2.contourArea(c)
                p = cv2.arcLength(c,True)
                tot_are+=area
                tot_per+=p
                
            
            num_cont.append(len(filtered))
            area_cont.append(tot_are)
            per_cont.append(tot_per)
            
            
df['num_contours']=num_cont
df['area_contours']=area_cont
df['len_contours']=per_cont
print(df.head())

df.to_csv (r'/Users/sanjana/Desktop/IP Paper/tonguedfff.csv', index = False, header=True)



