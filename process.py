# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:24:30 2019

@author: CYW
"""
import glob
import os
import shutil
import cv2
from xml.etree import ElementTree as ET


with open(r'F:\YOLO\keras-yolo3-master\data\valid.txt','r') as f:
    valid_imgs = list(map(lambda x:os.path.basename(x),f.read().split('\n')))

scale = 0.5

dir_s = glob.glob(r'F:\YOLO\keras-yolo3-master\data\label\*.xml')
for j,dir_ in enumerate(dir_s):
    
    img_path1 = os.path.join(r'F:\YOLO\keras-yolo3-master\data\img','%s.jpg' %os.path.basename(dir_)[:-4])
    img_path2 = os.path.join(r'F:\YOLO\keras-yolo3-master\data\img','%s.png' %os.path.basename(dir_)[:-4])
#    ubuntu_img_path = '/HDD/cyw/YOLO/keras-yolo3-master/data/img/%s.jpg' %os.path.basename(dir_)[:-4]
    
    flag =False
    if os.path.exists(img_path1):
        flag =True
        img_path = img_path1


    if os.path.exists(img_path2):
        flag =True
        img_path = img_path2

        
    if flag:    
        
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        #缩小图片
        size = (416, 416)
        shrink = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

            
        if '%s.jpg' %os.path.basename(dir_)[:-4] not in valid_imgs:
            set_name = 'train'
            ubuntu_img_path = '/HDD/cyw/YOLO/YOLOv3-model-pruning-master/data/images/train/%s.jpg' %os.path.basename(dir_)[:-4]

            cv2.imwrite(os.path.join(r'F:\YOLO\YOLOv3-model-pruning-master\data\images\train','%s.jpg' %os.path.basename(dir_)[:-4]),shrink)
            
                
            with open(r'F:\YOLO\YOLOv3-model-pruning-master\data\train.txt','a') as f:
                f.write('%s\n' %ubuntu_img_path)
    
        else:
            set_name = 'valid'
            ubuntu_img_path = '/HDD/cyw/YOLO/YOLOv3-model-pruning-master/data/images/valid/%s.jpg' %os.path.basename(dir_)[:-4]
            
            cv2.imwrite(os.path.join(r'F:\YOLO\YOLOv3-model-pruning-master\data\images\test','%s.jpg' %os.path.basename(dir_)[:-4]),shrink)  
            cv2.imwrite(os.path.join(r'F:\YOLO\YOLOv3-model-pruning-master\data\images\valid','%s.jpg' %os.path.basename(dir_)[:-4]),shrink) 
            
            with open(r'F:\YOLO\YOLOv3-model-pruning-master\data\valid.txt','a') as f:
                f.write('%s\n' %(ubuntu_img_path))



        
        tree = ET.parse(dir_)
        root = tree.getroot()
        jpg_width= int(root[4][0].text)
        jpg_height = int(root[4][1].text)
        
        length = len(root)
        filename = os.path.basename(dir_)[:-4]
        
        bbox_str = ''
        with open(r'F:\YOLO\YOLOv3-model-pruning-master\data\labels\%s\%s.txt' %(set_name,filename),'w') as f:
            for i in range(6,length):
                if root[i][0].text == 'Helmet':
                    label_idx = 0
                elif root[i][0].text == 'person':
                    label_idx = 1
                
                bbox = root[i][4]
                xmin = int(int(bbox[0].text))
                ymin = int(int(bbox[1].text))
                xmax = int(int(bbox[2].text))
                ymax = int(int(bbox[3].text))
                
                xmax = min(xmax, jpg_width-1)
                xmin = max(xmin, 0)
                ymax = min(ymax, jpg_height-1)
                ymin = max(ymin, 0)   
                
                
                norm_x_center = round((xmin + xmax)/2/jpg_width,4)
                norm_y_center = round((ymin + ymax)/2/jpg_height,4)
                norm_width = round((xmax - xmin)/jpg_width,4)
                norm_height = round((ymax - ymin)/jpg_height,4)
    
                if i != 6:
                    f.write("\n")            
                
                bbox_str = '%s %s %s %s %s' %(label_idx,norm_x_center,norm_y_center,norm_width,norm_height)
                f.write(bbox_str)
                


                
                
                
                
#%%              
                
                
                
                
                
                
                
                