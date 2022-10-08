from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree

import math

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vehicle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 428
    VALIDATION_STEPS = 80
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.92

    
config = CustomConfig()

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#Loading the model in the inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
# loading the trained weights o the custom dataset
model.load_weights("mask_rcnn_vehicle_0037.h5", by_name=True)

#%%
class_names = ['background', 'vehicle']

img = load_img("1d.jpg")
img = img_to_array(img)

# Run detection
results = model.detect([img], verbose=1)
 
# Visualize results
r = results[0]
bboxes = results[0]['rois']
masks = results[0]['masks']

img2 = load_img("2d.jpg")
img2 = img_to_array(img2)

# Run detection
results2 = model.detect([img2], verbose=1)
  
# Visualize results
r2 = results2[0]
bboxes2 = results2[0]['rois']
masks2 = results2[0]['masks']

visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'])

#%%

test_array = []
for i in range(bboxes2.shape[0]):
    test_array.append("")
    
    
    
for i in range(bboxes.shape[0]):
    if(abs((bboxes[i][2] - bboxes[i][0]) - (bboxes[i][3] - bboxes[i][1])) < 15):
        #çapraz    
        first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
        first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
            
        bbox_top_left_y = bboxes[i][0]
        bbox_top_left_x = bboxes[i][1]
        
        bbox_top_right_y = bboxes[i][0]
        bbox_top_right_x = bboxes[i][3]
        
        path = 0
        
        #choose path
        for k in range(20): #sol üstte mask var mı kontrol et. 20x20 piksel
            for l in range(20):
                
                if(masks[bbox_top_left_y + k][bbox_top_left_x + l][i]):
                    # print("Y= " + str(bbox_top_left_y + k) + " X= " + str(bbox_top_left_x + l))
                    path = 1
                    break
            
                else:
                    path = 0

        # masks list            
        masks_y_list = []
        masks_x_list = [] 
        for n in range(bboxes[i][0], bboxes[i][2]): #maskta true olanları x ve y listesine at
            for m in range(bboxes[i][1], bboxes[i][3]):
                if(masks[n][m][i]):
                    masks_y_list.append(n)
                    masks_x_list.append(m)                                         
        
        
        if(path == 1): #eğer sol üstte araba parçası varsaki yol
            top_right_point_index = masks_y_list.index(min(masks_y_list))
            top_right_y = masks_y_list[top_right_point_index]
            top_right_x = masks_x_list[top_right_point_index]
            
            top_left_point_index = masks_x_list.index(min(masks_x_list))
            top_left_y = masks_y_list[top_left_point_index]
            top_left_x = masks_x_list[top_left_point_index]
                        
            print("En sağ üst= " + str(top_right_y) + " " + str(top_right_x) + "    En sol üst= " + str(top_left_y) + " " + str(top_left_x))
            
            middle_y = (top_right_y + top_left_y) / 2
            middle_x = (top_right_x + top_left_x) / 2       
            
            hipotenus = math.sqrt(((first_image_center_x - middle_x)*(first_image_center_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
            karsi_kenar = math.sqrt(((middle_x - middle_x)*(middle_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
            
            angle = math.degrees(np.arcsin(karsi_kenar/hipotenus))
            
            print("Sola Yatık")
            print(math.degrees(np.arcsin(karsi_kenar/hipotenus)))
                   
            direction = "sol"
                       
        else: #eğer sağ üstte araba parçası varsaki yol
            top_left_point_index = masks_y_list.index(min(masks_y_list))
            top_left_y = masks_y_list[top_left_point_index]
            top_left_x = masks_x_list[top_left_point_index]
            
            top_right_point_index = masks_x_list.index(max(masks_x_list))
            top_right_y = masks_y_list[top_right_point_index]
            top_right_x = masks_x_list[top_right_point_index]
            
            print("En sol üst= " + str(top_left_y) + " " + str(top_left_x) + "    En sağ üst= " + str(top_right_y) + " " + str(top_right_x))
            
            middle_y = round((top_left_y + top_right_y) / 2)
            middle_x = round((top_left_x + top_right_x) / 2)       
            
            hipotenus = math.sqrt(((first_image_center_x - middle_x)*(first_image_center_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
            karsi_kenar = math.sqrt(((middle_x - middle_x)*(middle_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
            
            angle = math.degrees(np.arcsin(karsi_kenar/hipotenus))
            
            print("Sağa Yatık")
            print(math.degrees(np.arcsin(karsi_kenar/hipotenus)))       
                    
            direction = "sag"
        
        
        if(direction == "sol"): #üst taraf sola bakıyorsa           
            if(angle > 50 and angle <= 90):               
                for j in range(bboxes2.shape[0]):
                    second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                    second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                    
                    if(abs(first_image_center_y - second_image_center_y) < 50 and abs(first_image_center_x - second_image_center_x) < 30):                        
                        if(abs(first_image_center_y - second_image_center_y) < 7):
                            # print("sabit")
                            test_array[j] = "sabit"
                        elif(first_image_center_y >= second_image_center_y):
                            # print("yukari")
                            # test_array[j] = "yukari"
                            # if(first_image_center_x >= second_image_center_x):
                            test_array[j] = "sol yukari  Angle: " + str(round(angle))
                            # else:
                            #     test_array[j] = "sag yukari"
                            
                        elif(first_image_center_y < second_image_center_y):
                            # print("asagi")
                            # test_array[j] = "asagi"  
                            # if(first_image_center_x >= second_image_center_x):
                            #     test_array[j] = "sol asagi"
                            # else:
                            test_array[j] = "sag asagi  Angle: " + str(round(angle))
            
                    else:
                        # print("alakasız")
                        if(test_array[j] == ""):
                            test_array[j] = "alakasiz"
                            
                            
            elif(angle >= 40 and angle <= 50):
                for j in range(bboxes2.shape[0]):
                    second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                    second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                    
                    if(abs(first_image_center_y - second_image_center_y) < 80 and abs(first_image_center_x - second_image_center_x) < 80):                        
                        if(abs(first_image_center_y - second_image_center_y) < 7):
                            # print("sabit")
                            test_array[j] = "sabit"
                        elif(first_image_center_y >= second_image_center_y):
                            # print("yukari")
                            # test_array[j] = "yukari"
                            # if(first_image_center_x >= second_image_center_x):
                            test_array[j] = "sol yukari  Angle: " + str(round(angle))
                            # else:
                            #     test_array[j] = "sag yukari"
                            
                        elif(first_image_center_y < second_image_center_y):
                            # print("asagi")
                            # test_array[j] = "asagi"  
                            # if(first_image_center_x >= second_image_center_x):
                            #     test_array[j] = "sol asagi"
                            # else:
                            test_array[j] = "sag asagi  Angle: " + str(round(angle))
                            
                            
            elif(angle >= 0 and angle < 40):
                for j in range(bboxes2.shape[0]):
                    second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                    second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                    
                    if(abs(first_image_center_x - second_image_center_x) < 50 and abs(first_image_center_y - second_image_center_y) < 30):                        
                        if(abs(first_image_center_x - second_image_center_x) < 7):
                            # print("sabit")
                            test_array[j] = "sabit"
                        elif(first_image_center_x >= second_image_center_x):
                            # print("yukari")
                            # test_array[j] = "yukari"
                            # if(first_image_center_x >= second_image_center_x):
                            test_array[j] = "sol yukari  Angle: " + str(round(angle))
                            # else:
                            #     test_array[j] = "sag yukari"
                            
                        elif(first_image_center_x < second_image_center_x):
                            # print("asagi")
                            # test_array[j] = "asagi"  
                            # if(first_image_center_x >= second_image_center_x):
                            #     test_array[j] = "sol asagi"
                            # else:
                            test_array[j] = "sag asagi  Angle: " + str(round(angle))
                            
        if(direction == "sag"): #üst taraf sağa bakıyorsa
            if(angle > 50 and angle <= 90):               
                for j in range(bboxes2.shape[0]):
                    second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                    second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                    if(abs(first_image_center_y - second_image_center_y) < 50 and abs(first_image_center_x - second_image_center_x) < 30):                        
                        if(abs(first_image_center_y - second_image_center_y) < 7):
                            # print("sabit")
                            test_array[j] = "sabit"
                        elif(first_image_center_y >= second_image_center_y):
                            # print("yukari")
                            # test_array[j] = "yukari"
                            # if(first_image_center_x >= second_image_center_x):
                            test_array[j] = "sağ yukari  Angle: " + str(round(angle))
                            # else:
                            #     test_array[j] = "sag yukari"
                            
                        elif(first_image_center_y < second_image_center_y):
                            # print("asagi")
                            # test_array[j] = "asagi"  
                            # if(first_image_center_x >= second_image_center_x):
                            #     test_array[j] = "sol asagi"
                            # else:
                            test_array[j] = "sol asagi  Angle: " + str(round(angle))
                            
            elif(angle >= 40 and angle <= 50 ):
                for j in range(bboxes2.shape[0]):
                    second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                    second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                    if(abs(first_image_center_y - second_image_center_y) < 40 and abs(first_image_center_x - second_image_center_x) < 40):                        
                        if(abs(first_image_center_y - second_image_center_y) < 7):
                            # print("sabit")
                            test_array[j] = "sabit"
                        elif(first_image_center_y >= second_image_center_y):
                            # print("yukari")
                            # test_array[j] = "yukari"
                            # if(first_image_center_x >= second_image_center_x):
                            test_array[j] = "sağ yukari  Angle: " + str(round(angle))
                            # else:
                            #     test_array[j] = "sag yukari"
                            
                        elif(first_image_center_y < second_image_center_y):
                            # print("asagi")
                            # test_array[j] = "asagi"  
                            # if(first_image_center_x >= second_image_center_x):
                            #     test_array[j] = "sol asagi"
                            # else:
                            test_array[j] = "sol asagi  Angle: " + str(round(angle))
                                             
            elif(angle >= 0 and angle < 40):               
                for j in range(bboxes2.shape[0]):
                    second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                    second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                    if(abs(first_image_center_x - second_image_center_x) < 50 and abs(first_image_center_y - second_image_center_y) < 30):                        
                        if(abs(first_image_center_x - second_image_center_x) < 7):
                            # print("sabit")
                            test_array[j] = "sabit"
                        elif(first_image_center_x >= second_image_center_x):
                            # print("yukari")
                            # test_array[j] = "yukari"
                            # if(first_image_center_x >= second_image_center_x):
                            test_array[j] = "sağ yukari  Angle: " + str(round(angle))
                            # else:
                            #     test_array[j] = "sag yukari"
                            
                        elif(first_image_center_x < second_image_center_x):
                            # print("asagi")
                            # test_array[j] = "asagi"  
                            # if(first_image_center_x >= second_image_center_x):
                            #     test_array[j] = "sol asagi"
                            # else:
                            test_array[j] = "sol asagi  Angle: " + str(round(angle))
            
            
              
    elif(bboxes[i][2] - bboxes[i][0] > bboxes[i][3] - bboxes[i][1]): #dikey
        #dikey
        first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
        first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
        
        for j in range(bboxes2.shape[0]):
            second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
            second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
            
            if(abs(first_image_center_y - second_image_center_y) < 60 and abs(first_image_center_x - second_image_center_x) < 20):
                
                if(abs(first_image_center_y - second_image_center_y) < 10):
                    # print("sabit")
                    test_array[j] = "sabit"
                elif(first_image_center_y >= second_image_center_y):
                    # print("yukari")
                    test_array[j] = "yukari"
                elif(first_image_center_y < second_image_center_y):
                    # print("asagi")
                    test_array[j] = "asagi"            
    
            else:
                # print("alakasız")
                if(test_array[j] == ""):
                    test_array[j] = "alakasiz"        
    
    
    
    else: #yatay
        #yatay
        first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
        first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
        
        for j in range(bboxes2.shape[0]):
            second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
            second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
            
            if(abs(first_image_center_x - second_image_center_x) < 60 and abs(first_image_center_y - second_image_center_y) < 20):
                
                if(abs(first_image_center_x - second_image_center_x) < 10):
                    # print("sabit")
                    test_array[j] = "sabit"
                elif(first_image_center_x >= second_image_center_x):
                    # print("sola")
                    test_array[j] = "sola"
                elif(first_image_center_x < second_image_center_x):
                    # print("saga")
                    test_array[j] = "saga"            
    
            else:
                # print("alakasız")
                if(test_array[j] == ""):
                    test_array[j] = "alakasiz"


visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = test_array)



#%%
#yatay bulma
# for i in range(bboxes.shape[0]):
#     first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
#     first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
    
#     for j in range(bboxes2.shape[0]):
#         second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
#         second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
        
#         if(abs(first_image_center_x - second_image_center_x) < 60 and abs(first_image_center_y - second_image_center_y) < 20):
            
#             if(abs(first_image_center_x - second_image_center_x) < 10):
#                 print("sabit")
#                 test_array[j] = "sabit"
#             elif(first_image_center_x >= second_image_center_x):
#                 print("sola")
#                 test_array[j] = "sola"
#             elif(first_image_center_x < second_image_center_x):
#                 print("saga")
#                 test_array[j] = "saga"            

#         else:
#             print("alakasız")
#             if(test_array[j] == ""):
#                 test_array[j] = "alakasiz"

# visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = test_array)


#%%
#dikey
# test_array = []
# for i in range(bboxes2.shape[0]):
#     test_array.append("")

# for i in range(bboxes.shape[0]):
#     first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
#     first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
    
#     for j in range(bboxes2.shape[0]):
#         second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
#         second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
        
#         if(abs(first_image_center_y - second_image_center_y) < 60 and abs(first_image_center_x - second_image_center_x) < 20):
            
#             if(abs(first_image_center_y - second_image_center_y) < 10):
#                 print("sabit")
#                 test_array[j] = "sabit"
#             elif(first_image_center_y >= second_image_center_y):
#                 print("yukari")
#                 test_array[j] = "yukari"
#             elif(first_image_center_y < second_image_center_y):
#                 print("asagi")
#                 test_array[j] = "asagi"            

#         else:
#             print("alakasız")
#             if(test_array[j] == ""):
#                 test_array[j] = "alakasiz"

# visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = test_array)

#%%
# #bbox yönü gibi gibi
# test_array = []
# test = ""

# for i in range(bboxes.shape[0]):
#     for j in range(bboxes2.shape[0]):
#         if(bboxes[i][0] in range(bboxes2[j][0],bboxes2[j][2]) and bboxes[i][1] in range(bboxes2[j][1],bboxes2[j][3])):
#             test = "SolY"
#             print("sol veya yukarı")
#             break
#         elif(bboxes[i][2] in range(bboxes2[j][0],bboxes2[j][2]) and bboxes[i][3] in range(bboxes2[j][1],bboxes2[j][3])):
#             test = "SağA"
#             print("sağ veya aşağı")
#             break
#         elif(bboxes[i][0] in range(bboxes2[j][0],bboxes2[j][2]) and bboxes[i][3] in range(bboxes2[j][1],bboxes2[j][3])):
#             test = "SağY"
#             print("sağ veya yukarı")
#             break
#         elif(bboxes[i][2] in range(bboxes2[j][0],bboxes2[j][2]) and bboxes[i][1] in range(bboxes2[j][1],bboxes2[j][3])):
#             test = "SolA"
#             print("sol veya aşağı")
#             break
#         else:
#             test = "Sabit"
#             print("sabit")    
#     test_array.append(test)
#visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = test_array)
#%%

# all mask points
# class_names = ['background', 'vehicle']

# img = load_img("C:\\Users\\MONSTER\\Desktop\\b2.jpg")
# img = img_to_array(img)

# results = model.detect([img], verbose=1)

# mask_points = results[0]['masks']
# for k in range (mask_points.shape[2]):
#     for i in range(mask_points.shape[0]):
#         for j in range(mask_points.shape[1]):
#             if(mask_points[i][j][k]):
                
#%%
#araba duruşu
# img = load_img("C:\\Users\\MONSTER\\Desktop\\a.jpg")
# img = img_to_array(img)

# # Run detection
# results = model.detect([img], verbose=1)
# r = results[0]

# class_names = ['background', 'vehicle']

# bboxes = results[0]['rois']
# test = ""
# test_array = []
# for b in bboxes:
#     if(abs((b[2] - b[0]) - (b[3] - b[1])) < 13):
#         print("araba çarpraz duruyor")    
#         test = "araba çarpraz duruyor"
#     elif(b[2] - b[0] > b[3] - b[1]):
#         print("araba dikey duruyor")
#         test = "araba dikey duruyor"
#     else:
#         print("araba yatay duruyor")    
#         test = "araba yatay duruyor"
        
#     test_array.append(test)
   
# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], captions = test_array)
#%%
#sadece bbox'a alma
# data = load_img("C:\\Users\\MONSTER\\Desktop\\a.jpg")
# data = img_to_array(img)

# results = model.detect([img], verbose=1)
# boxes = results[0]['rois']

# y1, x1, y2, x2 = boxes[0]

# width, height = x2 - x1, y2 - y1
# # create the shape
# rect = Rectangle((x1, y1), width, height, fill=False, color='red')

# # draw an image with detected objects
# def draw_image_with_boxes(filename, boxes_list):
#      # load the image
#      data = pyplot.imread("C:\\Users\\MONSTER\\Desktop\\a.jpg")
#      # plot the image
#      pyplot.imshow(data)
#      # get the context for drawing boxes
#      ax = pyplot.gca()
#      # plot each box
#      for box in boxes_list:
#           # get coordinates
#           y1, x1, y2, x2 = box
#           # calculate width and height of the box
#           width, height = x2 - x1, y2 - y1
#           # create the shape
#           rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#           # draw the box
#           ax.add_patch(rect)
#      # show the plot
#      pyplot.show()

# draw_image_with_boxes("C:\\Users\\MONSTER\\Desktop\\a.jpg", results[0]['rois'])