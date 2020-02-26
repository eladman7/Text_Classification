# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:43:28 2017

@author: B
"""

import cv2
import os
patchSize=320
patchNumber=0
folder='test/'
for filename in os.listdir(folder):
    print(folder+filename)
    page=cv2.imread(folder+filename,1)
    rows,cols,ch=page.shape
    for x in range(0,rows-patchSize,patchSize):
        for y in range(0,cols-patchSize,patchSize):
            patch=page[x:x+patchSize,y:y+patchSize]
            cv2.imwrite("p"+folder+filename[:-4]+"_patch"+str(patchNumber)+".png",patch)
            patchNumber=patchNumber+1
