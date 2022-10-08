#important parameter list

#angle_list
#speed_list
#avg_env
#bboxes - bboxes2
#connected_components
#direction_list - group_direction_list
#group_angle_list - group_avg_angle_list
#group_speed_list - group_avg_speed_list
#traffic_congestion_list
#%%

# Python program to print connected  
# components in an undirected graph 
class Graph: 
      
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)] 
  
    def DFSUtil(self, temp, v, visited): 
  
        # Mark the current vertex as visited 
        visited[v] = True
  
        # Store the vertex to list 
        temp.append(v) 
  
        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 
  
    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 
  
    # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        connected_components = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                connected_components.append(self.DFSUtil(temp, v, visited)) 
        return connected_components 

def Pixel_To_Real_Speed(pixel):
    speed = pixel*2*5*60*60*0.00001
    return speed

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
model.load_weights("C:\\Users\\arasu\\Desktop\\otherLaptop\\transfer\\Mask_RCNN-final\\logs\\vehicle20200421T1417\\mask_rcnn_vehicle_0037.h5", by_name=True)
#img = load_img("C:\\Users\\MONSTER\\Desktop\\transfer\\Mask_RCNN\\test.jpg")
#img = load_img("C:\\Users\\MONSTER\\Desktop\\SonVeriseti\\SonVeriseti\\train\\Vehicle\\vehicle0157.png")
#img = img_to_array(img)

#%%
specimen = 1
for z in range(0, 15):
    
    image_path1 = "C:\\Users\\arasu\\Desktop\\scenario3\\ScreenShot" + str(specimen) + ".png"
    image_path2 = "C:\\Users\\arasu\\Desktop\\scenario3\\ScreenShot" + str(specimen+1) + ".png"
    output_path = "C:\\Users\\arasu\\Desktop\\scenario3\\Output" + str(specimen) + "-" + str(specimen+1) + ".png"
    specimen = specimen + 2
    
    class_names = ['background', 'vehicle']
    
    # img = load_img("C:\\Users\\arasu\\Desktop\\otherLaptop\\1a.png")
    img = load_img(image_path1)
    img = img_to_array(img)
    
    # Run detection
    results = model.detect([img], verbose=1)
     
    # Visualize results
    r = results[0]
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    bboxes = results[0]['rois']
    masks = results[0]['masks']
    
    # img2 = load_img("C:\\Users\\arasu\\Desktop\\otherLaptop\\1b.png")
    img2 = load_img(image_path2)
    img2 = img_to_array(img2)
    
    # Run detection
    results2 = model.detect([img2], verbose=1)
      
    # Visualize results
    r2 = results2[0]
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    bboxes2 = results2[0]['rois']
    masks2 = results2[0]['masks']
    
    empty_captions = []
    for a in range(len(bboxes)):
        empty_captions.append("")
        
    empty_captions2 = []
    for a in range(len(bboxes2)):
        empty_captions2.append("")
        
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], captions = empty_captions)
    visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = empty_captions2)
    
    
    #average bbox env.
    sum_x = 0
    sum_y = 0
    for i in range(bboxes.shape[0]):
        sum_y = sum_y + int(round(((bboxes[i][2] - bboxes[i][0]))))
        sum_x = sum_x + int(round(((bboxes[i][3] - bboxes[i][1]))))
                            
    avg_y = int(round((sum_y/bboxes.shape[0])*2))
    avg_x = int(round((sum_x/bboxes.shape[0])*2))
    
    
    sum_of_env = 0
    for i in range(bboxes.shape[0]):
        if(int(round((bboxes[i][2] - bboxes[i][0])))*2 <= avg_y and int(round((bboxes[i][3] - bboxes[i][1])))*2 <= avg_x):
            sum_of_env =sum_of_env + int(round(((bboxes[i][2] - bboxes[i][0]) + (bboxes[i][3] - bboxes[i][1]))))
    
    avg_env = int(round((sum_of_env/bboxes.shape[0])*2))
    
    
    #direction calculation
    
    direction_list = []
    for i in range(bboxes2.shape[0]):
        direction_list.append("Irrelevant")
               
    for i in range(bboxes.shape[0]):
        if(abs((bboxes[i][2] - bboxes[i][0]) - (bboxes[i][3] - bboxes[i][1])) < int(round(avg_env/100*14))): #14 degistim
            #çapraz    
            first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
            first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
                
            bbox_top_left_y = bboxes[i][0]
            bbox_top_left_x = bboxes[i][1]
            
            bbox_top_right_y = bboxes[i][0]
            bbox_top_right_x = bboxes[i][3]
            
            path = 0
            
            #choose path
            for k in range(int(round(avg_env/100*4))): #sol üstte mask var mı kontrol et. 20x20 piksel
                for l in range(int(round(avg_env/100*4))):
                    
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
                                                              
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*18)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*7))):
                            if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*2.7))):
                                # print("Constant")
                                direction_list[j] = "Constant"
                            elif(first_image_center_y >= second_image_center_y):
                                # print("yukari")
                                # direction_list[j] = "yukari"
                                # if(first_image_center_x >= second_image_center_x):
                                direction_list[j] = "Left Up"
                                # else:
                                #     direction_list[j] = "sag yukari"
                                
                            elif(first_image_center_y < second_image_center_y):
                                # print("asagi")
                                # direction_list[j] = "asagi"  
                                # if(first_image_center_x >= second_image_center_x):
                                #     direction_list[j] = "sol asagi"
                                # else:
                                direction_list[j] = "Right Down"
                                
                        else:
                            # print("Irrelevant")
                            if(direction_list[j] == ""):
                                direction_list[j] = "Irrelevant"
                                
                                
                elif(angle >= 40 and angle <= 50):
                    for j in range(bboxes2.shape[0]):
                        second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                        second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                                         
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*14)) and abs(first_image_center_x - second_image_center_x) < int(round(((bboxes[i][2] - bboxes[i][0]) + (bboxes[i][3] - bboxes[i][1]))*2/100*14))):
                            if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*2.2))):
                                # print("Constant")
                                direction_list[j] = "Constant"
                            elif(first_image_center_y >= second_image_center_y):
                                # print("yukari")
                                # direction_list[j] = "yukari"
                                # if(first_image_center_x >= second_image_center_x):
                                direction_list[j] = "Left Up"
                                # else:
                                #     direction_list[j] = "sag yukari"
                                
                            elif(first_image_center_y < second_image_center_y):
                                # print("asagi")
                                # direction_list[j] = "asagi"  
                                # if(first_image_center_x >= second_image_center_x):
                                #     direction_list[j] = "sol asagi"
                                # else:
                                direction_list[j] = "Right Down"                          
                                
                elif(angle >= 0 and angle < 40):
                    for j in range(bboxes2.shape[0]):
                        second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                        second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                        
                        if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*18)) and abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*7))):                        
                            if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*2.7))):
                                # print("Constant")
                                direction_list[j] = "Constant"
                            elif(first_image_center_x >= second_image_center_x):
                                # print("yukari")
                                # direction_list[j] = "yukari"
                                # if(first_image_center_x >= second_image_center_x):
                                direction_list[j] = "Left Up"
                                # else:
                                #     direction_list[j] = "sag yukari"
                                
                            elif(first_image_center_x < second_image_center_x):
                                # print("asagi")
                                # direction_list[j] = "asagi"  
                                # if(first_image_center_x >= second_image_center_x):
                                #     direction_list[j] = "sol asagi"
                                # else:
                                direction_list[j] = "Right Down"
                                
            if(direction == "sag"): #üst taraf sağa bakıyorsa
                if(angle > 50 and angle <= 90):               
                    for j in range(bboxes2.shape[0]):
                        second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                        second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                        
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*18)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*7))):                        
                            if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*2.7))):
                                # print("Constant")
                                direction_list[j] = "Constant"
                            elif(first_image_center_y >= second_image_center_y):
                                # print("yukari")
                                # direction_list[j] = "yukari"
                                # if(first_image_center_x >= second_image_center_x):
                                direction_list[j] = "Right Up"
                                # else:
                                #     direction_list[j] = "sag yukari"
                                
                            elif(first_image_center_y < second_image_center_y):
                                # print("asagi")
                                # direction_list[j] = "asagi"  
                                # if(first_image_center_x >= second_image_center_x):
                                #     direction_list[j] = "sol asagi"
                                # else:
                                direction_list[j] = "Left Down"
                                
                elif(angle >= 40 and angle <= 50 ):
                    for j in range(bboxes2.shape[0]):
                        second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                        second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                        
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*14)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*14))):                                       
                            if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*2.2))):
                                # print("Constant")
                                direction_list[j] = "Constant"
                            elif(first_image_center_y >= second_image_center_y):
                                # print("yukari")
                                # direction_list[j] = "yukari"
                                # if(first_image_center_x >= second_image_center_x):
                                direction_list[j] = "Right Up"
                                # else:
                                #     direction_list[j] = "sag yukari"
                                
                            elif(first_image_center_y < second_image_center_y):
                                # print("asagi")
                                # direction_list[j] = "asagi"  
                                # if(first_image_center_x >= second_image_center_x):
                                #     direction_list[j] = "sol asagi"
                                # else:
                                direction_list[j] = "Left Down"
                                                 
                elif(angle >= 0 and angle < 40):               
                    for j in range(bboxes2.shape[0]):
                        second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                        second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                        
                        if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*18)) and abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*7))):                        
                            if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*2.7))):
                                # print("Constant")
                                direction_list[j] = "Constant"
                            elif(first_image_center_x >= second_image_center_x):
                                # print("yukari")
                                # direction_list[j] = "yukari"
                                # if(first_image_center_x >= second_image_center_x):
                                direction_list[j] = "Right Up"
                                # else:
                                #     direction_list[j] = "sag yukari"
                                
                            elif(first_image_center_x < second_image_center_x):
                                # print("asagi")
                                # direction_list[j] = "asagi"  
                                # if(first_image_center_x >= second_image_center_x):
                                #     direction_list[j] = "sol asagi"
                                # else:
                                direction_list[j] = "Left Down"
                
                
                  
        elif(bboxes[i][2] - bboxes[i][0] > bboxes[i][3] - bboxes[i][1]): #dikey
            #dikey
                   
            first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
            first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
            
            for j in range(bboxes2.shape[0]):
                second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                
                if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*19)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*7))):
                    if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*3))):
                        # print("Constant")
                        direction_list[j] = "Constant"
                    elif(first_image_center_y >= second_image_center_y):
                        # print("yukari")
                        direction_list[j] = "Directly Up"
                    elif(first_image_center_y < second_image_center_y):
                        # print("asagi")
                        direction_list[j] = "Directly Down"            
        
                else:
                    # print("Irrelevant")
                    if(direction_list[j] == ""):
                        direction_list[j] = "Irrelevant"        
        
        
        
        else: #yatay
            #yatay
            
            
            first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
            first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
            
            for j in range(bboxes2.shape[0]):
                second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
    
                if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*19)) and abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*7))):
                    if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*3))):
                        # print("Constant")
                        direction_list[j] = "Constant"
                    elif(first_image_center_x >= second_image_center_x):
                        # print("sola")
                        direction_list[j] = "Directly Left"
                    elif(first_image_center_x < second_image_center_x):
                        # print("saga")
                        direction_list[j] = "Directly Right"            
        
                else:
                    # print("Irrelevant")
                    if(direction_list[j] == ""):
                        direction_list[j] = "Irrelevant"
    
    
    visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = direction_list)
    
    #second image angle list
    
    angle_list = []
           
    for i in range(bboxes2.shape[0]):
        if(abs((bboxes2[i][2] - bboxes2[i][0]) - (bboxes2[i][3] - bboxes2[i][1])) < int(round(avg_env/100*14))): #degistim
            #çapraz    
            first_image_center_y = round(bboxes2[i][0] + ((bboxes2[i][2] - bboxes2[i][0])/2))
            first_image_center_x = round(bboxes2[i][1] + ((bboxes2[i][3] - bboxes2[i][1])/2))
                
            bbox_top_left_y = bboxes2[i][0]
            bbox_top_left_x = bboxes2[i][1]
            
            bbox_top_right_y = bboxes2[i][0]
            bbox_top_right_x = bboxes2[i][3]
            
            path = 0
            
            #choose path
            for k in range(int(round(avg_env/100*4))): #sol üstte mask var mı kontrol et. 20x20 piksel
                for l in range(int(round(avg_env/100*4))):
                    
                    if(masks2[bbox_top_left_y + k][bbox_top_left_x + l][i]):
                        # print("Y= " + str(bbox_top_left_y + k) + " X= " + str(bbox_top_left_x + l))
                        path = 1
                        break
                
                    else:
                        path = 0
    
            # masks2 list            
            masks2_y_list = []
            masks2_x_list = [] 
            for n in range(bboxes2[i][0], bboxes2[i][2]): #maskta true olanları x ve y listesine at
                for m in range(bboxes2[i][1], bboxes2[i][3]):
                    if(masks2[n][m][i]):
                        masks2_y_list.append(n)
                        masks2_x_list.append(m)                                         
            
            
            if(path == 1): #eğer sol üstte araba parçası varsaki yol
                top_right_point_index = masks2_y_list.index(min(masks2_y_list))
                top_right_y = masks2_y_list[top_right_point_index]
                top_right_x = masks2_x_list[top_right_point_index]
                
                top_left_point_index = masks2_x_list.index(min(masks2_x_list))
                top_left_y = masks2_y_list[top_left_point_index]
                top_left_x = masks2_x_list[top_left_point_index]
                            
                print("En sağ üst= " + str(top_right_y) + " " + str(top_right_x) + "    En sol üst= " + str(top_left_y) + " " + str(top_left_x))
                
                middle_y = (top_right_y + top_left_y) / 2
                middle_x = (top_right_x + top_left_x) / 2       
                
                hipotenus = math.sqrt(((first_image_center_x - middle_x)*(first_image_center_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
                karsi_kenar = math.sqrt(((middle_x - middle_x)*(middle_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
                
                angle = math.degrees(np.arcsin(karsi_kenar/hipotenus))
                
                print(math.degrees(np.arcsin(karsi_kenar/hipotenus)))
                       
                direction = "sol"
                           
            else: #eğer sağ üstte araba parçası varsaki yol
                top_left_point_index = masks2_y_list.index(min(masks2_y_list))
                top_left_y = masks2_y_list[top_left_point_index]
                top_left_x = masks2_x_list[top_left_point_index]
                
                top_right_point_index = masks2_x_list.index(max(masks2_x_list))
                top_right_y = masks2_y_list[top_right_point_index]
                top_right_x = masks2_x_list[top_right_point_index]
                
                print("En sol üst= " + str(top_left_y) + " " + str(top_left_x) + "    En sağ üst= " + str(top_right_y) + " " + str(top_right_x))
                
                middle_y = round((top_left_y + top_right_y) / 2)
                middle_x = round((top_left_x + top_right_x) / 2)       
                
                hipotenus = math.sqrt(((first_image_center_x - middle_x)*(first_image_center_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
                karsi_kenar = math.sqrt(((middle_x - middle_x)*(middle_x - middle_x)) + ((first_image_center_y - middle_y)*(first_image_center_y - middle_y)))
                
                angle = math.degrees(np.arcsin(karsi_kenar/hipotenus))
                
                print(math.degrees(np.arcsin(karsi_kenar/hipotenus)))       
                        
                direction = "sag"
            
            angle_list.append(int(round(angle)))
            
                  
        elif(bboxes2[i][2] - bboxes2[i][0] > bboxes2[i][3] - bboxes2[i][1]): #dikey
            #dikey        
            angle_list.append(90)
                     
        
        else: #yatay
            angle_list.append(0)
    
    visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = angle_list)
    
    #second image speed list
    
    speed_list = []
    for i in range(bboxes2.shape[0]):
        speed_list.append(0)
               
    for i in range(bboxes2.shape[0]):
        if(abs((bboxes2[i][2] - bboxes2[i][0]) - (bboxes2[i][3] - bboxes2[i][1])) < int(round(avg_env/100*14))): #degistim
            #çapraz    
            first_image_center_y = round(bboxes2[i][0] + ((bboxes2[i][2] - bboxes2[i][0])/2))
            first_image_center_x = round(bboxes2[i][1] + ((bboxes2[i][3] - bboxes2[i][1])/2))
                
            bbox_top_left_y = bboxes2[i][0]
            bbox_top_left_x = bboxes2[i][1]
            
            bbox_top_right_y = bboxes2[i][0]
            bbox_top_right_x = bboxes2[i][3]
            
            path = 0
            
            #choose path
            for k in range(int(round(avg_env/100*4))): #sol üstte mask var mı kontrol et. 20x20 piksel
                for l in range(int(round(avg_env/100*4))):
                    
                    if(masks2[bbox_top_left_y + k][bbox_top_left_x + l][i]):
                        # print("Y= " + str(bbox_top_left_y + k) + " X= " + str(bbox_top_left_x + l))
                        path = 1
                        break
                
                    else:
                        path = 0
    
            # masks2 list            
            masks2_y_list = []
            masks2_x_list = [] 
            for n in range(bboxes2[i][0], bboxes2[i][2]): #maskta true olanları x ve y listesine at
                for m in range(bboxes2[i][1], bboxes2[i][3]):
                    if(masks2[n][m][i]):
                        masks2_y_list.append(n)
                        masks2_x_list.append(m)                                         
            
            
            if(path == 1): #eğer sol üstte araba parçası varsaki yol
                top_right_point_index = masks2_y_list.index(min(masks2_y_list))
                top_right_y = masks2_y_list[top_right_point_index]
                top_right_x = masks2_x_list[top_right_point_index]
                
                top_left_point_index = masks2_x_list.index(min(masks2_x_list))
                top_left_y = masks2_y_list[top_left_point_index]
                top_left_x = masks2_x_list[top_left_point_index]
                            
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
                top_left_point_index = masks2_y_list.index(min(masks2_y_list))
                top_left_y = masks2_y_list[top_left_point_index]
                top_left_x = masks2_x_list[top_left_point_index]
                
                top_right_point_index = masks2_x_list.index(max(masks2_x_list))
                top_right_y = masks2_y_list[top_right_point_index]
                top_right_x = masks2_x_list[top_right_point_index]
                
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
                    for j in range(bboxes.shape[0]):
                        second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                        second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                                                              
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*18)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*7))):                        
                            speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                            speed_list[i] = round(speed)
                                
                                
                elif(angle >= 40 and angle <= 50):
                    for j in range(bboxes.shape[0]):
                        second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                        second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                                         
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*14)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*14))):                        
                            speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                            speed_list[i] = round(speed)               
                                
                elif(angle >= 0 and angle < 40):
                    for j in range(bboxes.shape[0]):
                        second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                        second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                        
                        if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*18)) and abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*7))):                        
                            speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                            speed_list[i] = round(speed)
                                
            if(direction == "sag"): #üst taraf sağa bakıyorsa
                if(angle > 50 and angle <= 90):               
                    for j in range(bboxes.shape[0]):
                        second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                        second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                        
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*18)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*7))):                        
                            speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                            speed_list[i] = round(speed)
    
                elif(angle >= 40 and angle <= 50 ):
                    for j in range(bboxes.shape[0]):
                        second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                        second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                        
                        if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*14)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*14))):                        
                            speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                            speed_list[i] = round(speed)                       
                                                 
                elif(angle >= 0 and angle < 40):               
                    for j in range(bboxes.shape[0]):
                        second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                        second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                        
                        if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*18)) and abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*7))):                        
                            speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                            speed_list[i] = round(speed)
                                      
                  
        elif(bboxes2[i][2] - bboxes2[i][0] > bboxes2[i][3] - bboxes2[i][1]): #dikey
            #dikey
                   
            first_image_center_y = round(bboxes2[i][0] + ((bboxes2[i][2] - bboxes2[i][0])/2))
            first_image_center_x = round(bboxes2[i][1] + ((bboxes2[i][3] - bboxes2[i][1])/2))
            
            for j in range(bboxes.shape[0]):
                second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
                
                if(abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*19)) and abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*7))):
                    speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                    speed_list[i] = round(speed)
                        
        else: #yatay
            #yatay
            
            
            first_image_center_y = round(bboxes2[i][0] + ((bboxes2[i][2] - bboxes2[i][0])/2))
            first_image_center_x = round(bboxes2[i][1] + ((bboxes2[i][3] - bboxes2[i][1])/2))
            
            for j in range(bboxes.shape[0]):
                second_image_center_y = round(bboxes[j][0] + ((bboxes[j][2] - bboxes[j][0])/2))
                second_image_center_x = round(bboxes[j][1] + ((bboxes[j][3] - bboxes[j][1])/2))
    
                if(abs(first_image_center_x - second_image_center_x) < int(round(avg_env/100*19)) and abs(first_image_center_y - second_image_center_y) < int(round(avg_env/100*7))):
                    speed = math.sqrt(((first_image_center_x - second_image_center_x)*(first_image_center_x - second_image_center_x)) + ((first_image_center_y - second_image_center_y)*(first_image_center_y - second_image_center_y)))
                    speed_list[i] = round(speed)
                    
    
    visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = speed_list)
    
    
    #graphs
    g = Graph(bboxes2.shape[0]);
    
    for i in range(bboxes2.shape[0]):
    #for i in range(1, 2):
            first_image_center_y = round(bboxes2[i][0] + ((bboxes2[i][2] - bboxes2[i][0])/2))
            first_image_center_x = round(bboxes2[i][1] + ((bboxes2[i][3] - bboxes2[i][1])/2))
            
            kare_kontrol_baslangic_y = first_image_center_y - (avg_env/100*30) #27
            kare_kontrol_baslangic_x = first_image_center_x - (avg_env/100*30)
                
            kare_kontrol_bitis_y = first_image_center_y + (avg_env/100*30)
            kare_kontrol_bitis_x = first_image_center_x + (avg_env/100*30)
                          
            for j in range (bboxes2.shape[0]):
                second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
                second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
                
                corner_1_y = bboxes2[j][0]
                corner_1_x = bboxes2[j][1]
            
                corner_2_y = bboxes2[j][0]
                corner_2_x = bboxes2[j][3]
                
                corner_3_y = bboxes2[j][2]
                corner_3_x = bboxes2[j][1]
                
                corner_4_y = bboxes2[j][2]
                corner_4_x = bboxes2[j][3]
                
                check_y = range(int(kare_kontrol_baslangic_y), int(kare_kontrol_bitis_y+1))
                check_x = range(int(kare_kontrol_baslangic_x), int(kare_kontrol_bitis_x+1))
                
                if(((corner_1_y in check_y and corner_1_x in check_x) or (corner_2_y in check_y and corner_2_x in check_x) or (corner_3_y in check_y and corner_3_x in check_x) or (corner_4_y in check_y and corner_4_x in check_x)) and i != j):
                    print("kontrol edilen = " + str(i) + "   " + "içinde duran = " + str(j))
                    g.addEdge(i, j)
            
    connected_components = g.connectedComponents()
    print("Following are connected components") 
    print(connected_components) 
    
    traffic_congestion_list = []
    group_bboxes = []
    group_direction_list = []
    group_avg_speed_list = []
    group_avg_real_speed_list = []
    group_avg_angle_list = []
    group_angle_inconsistency_list = []
    group_crowded_list = []
    group_score_list = []
    group_traffic_real_distance_list = []
    # serit_max_list = []
    
    for i in range(len(connected_components)):
        
        Directly_Down_count = 0
        Directly_Up_count = 0
        
        Left_Up_count = 0
        Left_Down_count = 0
        
        Right_Up_count = 0
        Right_Down_count = 0
        
        Constant_count = 0
        Irrelevant_count = 0
        
        group_y1 = img2.shape[0]
        group_x1 = img2.shape[1]
        group_y2 = 0
        group_x2 = 0
        
        group_speed = 0
        group_avg_speed = 0
        
        group_angle = 0
        group_avg_angle = 0
        group_angle_inconsistency = 0   
        
        for j in range(len(connected_components[i])):
            print(direction_list[connected_components[i][j]])
            
            #angle
            group_angle = group_angle + angle_list[connected_components[i][j]]
            
            #speed
            group_speed = group_speed + speed_list[connected_components[i][j]]
            
            #direction
            if(direction_list[connected_components[i][j]] == "Directly Down"):
                Directly_Down_count += 1
            elif(direction_list[connected_components[i][j]] == "Directly Up"):
                Directly_Up_count += 1
                
            elif(direction_list[connected_components[i][j]] == "Left Up"):
                Left_Up_count += 1
            elif(direction_list[connected_components[i][j]] == "Left Down"):
                Left_Down_count += 1
                
            elif(direction_list[connected_components[i][j]] == "Right Up"):
                Right_Up_count += 1
            elif(direction_list[connected_components[i][j]] == "Right Down"):
                Right_Down_count += 1
            
            elif(direction_list[connected_components[i][j]] == "Constant"):
                Constant_count += 1
            elif(direction_list[connected_components[i][j]] == "Irrelevant"):
                Irrelevant_count += 1
            #--     
            if(bboxes2[connected_components[i][j]][0] < group_y1):
                group_y1 = bboxes2[connected_components[i][j]][0]
            if(bboxes2[connected_components[i][j]][1] < group_x1):
                group_x1 = bboxes2[connected_components[i][j]][1]
            if(bboxes2[connected_components[i][j]][2] > group_y2):
                group_y2 = bboxes2[connected_components[i][j]][2]
            if(bboxes2[connected_components[i][j]][3] > group_x2):
                group_x2 = bboxes2[connected_components[i][j]][3]
            
        group_avg_angle = int(round(group_angle/len(connected_components[i])))
        group_avg_angle_list.append(group_avg_angle)
        
        for j in range(len(connected_components[i])):
            if(abs(group_avg_angle - angle_list[connected_components[i][j]]) > 5):
               group_angle_inconsistency =  group_angle_inconsistency + (abs(group_avg_angle - angle_list[connected_components[i][j]]))
        
        group_angle_inconsistency_list.append(group_angle_inconsistency/len(connected_components[i]))              
            
        group_avg_speed = int(round(group_speed/len(connected_components[i])))
        group_avg_speed_list.append(group_avg_speed)    
        group_avg_real_speed_list.append(Pixel_To_Real_Speed(group_avg_speed))
        # print(group_avg_speed)
        
        print(" " + str(group_y1) + " " + str(group_x1) + " " + str(group_y2) + " " + str(group_x2))  
        group_bboxes.append([group_y1, group_x1, group_y2, group_x2])
    
        if(abs(group_y1 - group_y2) >= abs(group_x1 - group_x2)):
            group_traffic_real_distance_list.append(abs(group_y1 - group_y2)*5/100)
        elif(abs(group_x1 - group_x2) > abs(group_y1 - group_y2)):
            group_traffic_real_distance_list.append(abs(group_x1 - group_x2)*5/100)
        
        l = [Directly_Down_count, Directly_Up_count, Left_Up_count, Left_Down_count, Right_Up_count, Right_Down_count]
        
        # serit_list = []
        
      
        # print(l.index(max(l)))
        if(l.index(max(l)) == 0):
            print("Aşağı Gidiyor")
            group_direction_list.append("Directly Down")
            # for j in range(len(connected_components[i])):
            #     serit = 0
            #     kontrol_x1 = bboxes2[connected_components[i][j]][1] - (avg_env/100*27)
            #     kontrol_y1 = bboxes2[connected_components[i][j]][0]
            #     kontrol_x2 = bboxes2[connected_components[i][j]][3] + (avg_env/100*27)
            #     kontrol_y2 = bboxes2[connected_components[i][j]][2]
            #     for k in range(len(connected_components[i])):              
            #         corner_1_y = bboxes2[connected_components[i][k]][0]
            #         corner_1_x = bboxes2[connected_components[i][k]][1]
                
            #         corner_2_y = bboxes2[connected_components[i][k]][0]
            #         corner_2_x = bboxes2[connected_components[i][k]][3]
                    
            #         corner_3_y = bboxes2[connected_components[i][k]][2]
            #         corner_3_x = bboxes2[connected_components[i][k]][1]
                    
            #         corner_4_y = bboxes2[connected_components[i][k]][2]
            #         corner_4_x = bboxes2[connected_components[i][k]][3]
                    
            #         check_y = range(int(kontrol_y1), int(kontrol_y2+1))
            #         check_x = range(int(kontrol_x1), int(kontrol_x2+1))
            #         if(((corner_1_y in check_y and corner_1_x in check_x) or (corner_2_y in check_y and corner_2_x in check_x) or (corner_3_y in check_y and corner_3_x in check_x) or (corner_4_y in check_y and corner_4_x in check_x))):      
            #             serit+=1
            #             serit_list.append(serit)
            #             # print(serit)
            # print("-----------")
            # serit_max_list.append(max(serit_list))
                                            
        elif((l.index(max(l)) == 1)):
            print("Yukari Gidiyor")
            group_direction_list.append("Directly Up")
            
        elif((l.index(max(l)) == 2)):
            print("Sol Yukari Gidiyor")
            group_direction_list.append("Left Up")  
        elif((l.index(max(l)) == 3)):
            print("Sol Asagi Gidiyor")
            group_direction_list.append("Left Down")   
        elif((l.index(max(l)) == 4)):
            print("Sag Yukari Gidiyor")
            group_direction_list.append("Right Up")  
        elif((l.index(max(l)) == 5)):
            print("Sag Asagi Gidiyor")
            group_direction_list.append("Right Down") 
        
                         
        if(len(connected_components[i]) >= 4):
            # if(Constant_count >= 5):
            traffic_congestion_list.append("Crowded")
            group_crowded_list.append(i)
        else:
            traffic_congestion_list.append("empty")
            
    
    
    for i in range(len(connected_components)):       
        ##score calculation
        
        if(group_angle_inconsistency_list[i] >= 0 and group_angle_inconsistency_list[i] <= 5):
            if(group_avg_real_speed_list[i] >= 45):
                group_score_list.append(0)
            elif(group_avg_real_speed_list[i] > 30 and group_avg_real_speed_list[i] < 45):
                group_score_list.append(1)
            elif(group_avg_real_speed_list[i] > 20 and group_avg_real_speed_list[i] <= 30):
                group_score_list.append(2)
            elif(group_avg_real_speed_list[i] > 10 and group_avg_real_speed_list[i] <= 20):
                group_score_list.append(3)
            elif(group_avg_real_speed_list[i] <= 10):
                group_score_list.append(4)
            else:
                group_score_list.append(-100)
                
        elif(group_angle_inconsistency_list[i] > 5 and group_angle_inconsistency_list[i] <= 10):
            if(group_avg_real_speed_list[i] >= 45):
                group_score_list.append(0)
            elif(group_avg_real_speed_list[i] > 30 and group_avg_real_speed_list[i] < 45):
                group_score_list.append(1)
            elif(group_avg_real_speed_list[i] > 20 and group_avg_real_speed_list[i] <= 30):
                group_score_list.append(2)
            elif(group_avg_real_speed_list[i] > 10 and group_avg_real_speed_list[i] <= 20):
                group_score_list.append(4)
            elif(group_avg_real_speed_list[i] <= 10):
                group_score_list.append(5)
            else:
                group_score_list.append(-100)
                
        elif(group_angle_inconsistency_list[i] > 10):
            if(group_avg_real_speed_list[i] >= 45):
                group_score_list.append(0)
            elif(group_avg_real_speed_list[i] > 30 and group_avg_real_speed_list[i] < 45):
                group_score_list.append(1)
            elif(group_avg_real_speed_list[i] > 20 and group_avg_real_speed_list[i] <= 30):
                group_score_list.append(2)
            elif(group_avg_real_speed_list[i] > 10 and group_avg_real_speed_list[i] <= 20):
                group_score_list.append(5)
            elif(group_avg_real_speed_list[i] <= 10):
                group_score_list.append(6)
            else:
                group_score_list.append(-100)
        else:
            group_score_list[i] = -100
    
    
    
    # image = cv2.imread("C:\\Users\\arasu\\Desktop\\otherLaptop\\1b.png") 
    image = cv2.imread(image_path2) 
    
    for i in range(len(group_crowded_list)):
        
        if(traffic_congestion_list[group_crowded_list[i]] == "Crowded"):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            
        cv2.rectangle(image,
                        (group_bboxes[group_crowded_list[i]][1], group_bboxes[group_crowded_list[i]][0]),
                        (group_bboxes[group_crowded_list[i]][3], group_bboxes[group_crowded_list[i]][2]),
                        color, thickness=2)
        info = traffic_congestion_list[group_crowded_list[i]] +"  "+ group_direction_list[group_crowded_list[i]] + "  " + str(round(group_avg_real_speed_list[group_crowded_list[i]],2)) + " km/h  " + str(round(group_traffic_real_distance_list[group_crowded_list[i]],2)) + " meter  " + str(group_score_list[group_crowded_list[i]]) + " score"
        # info = ""
        
        cv2.putText(image, info, (int(group_bboxes[group_crowded_list[i]][1])-200,int(group_bboxes[group_crowded_list[i]][0])+200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color, 1)
            
    # Saving the image  
    # cv2.imwrite("C:\\Users\\arasu\\Desktop\\output.png",image)
    cv2.imwrite(output_path,image)
    
    
    #%%
    
    # image = cv2.imread("C:\\Users\\arasu\\Desktop\\otherLaptop\\1b.png") 
    
    
    # for i in range(len(group_bboxes)):
        
    #     if(traffic_congestion_list[i] == "crowded"):
    #         color = (0, 0, 255)
    #     else:
    #         color = (0, 255, 0)
            
    #     cv2.rectangle(image,
    #                     (group_bboxes[i][1], group_bboxes[i][0]),
    #                     (group_bboxes[i][3], group_bboxes[i][2]),
    #                     color, thickness=2)
    #     info = traffic_congestion_list[i] +"  "+ group_direction_list[i]
    #     # info = ""
        
    #     cv2.putText(image, info, (int(group_bboxes[i][1])-20,int(group_bboxes[i][0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color, 1)
            
    # # Saving the image  
    # cv2.imwrite("C:\\Users\\arasu\\Desktop\\output.png",image)
    
    
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
    #                 print("Constant")
    #                 direction_list[j] = "Constant"
    #             elif(first_image_center_x >= second_image_center_x):
    #                 print("sola")
    #                 direction_list[j] = "sola"
    #             elif(first_image_center_x < second_image_center_x):
    #                 print("saga")
    #                 direction_list[j] = "saga"            
    
    #         else:
    #             print("Irrelevant")
    #             if(direction_list[j] == ""):
    #                 direction_list[j] = "Irrelevant"
    
    # visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = direction_list)
    
    
    #%%
    #dikey
    # direction_list = []
    # for i in range(bboxes2.shape[0]):
    #     direction_list.append("")
    
    # for i in range(bboxes.shape[0]):
    #     first_image_center_y = round(bboxes[i][0] + ((bboxes[i][2] - bboxes[i][0])/2))
    #     first_image_center_x = round(bboxes[i][1] + ((bboxes[i][3] - bboxes[i][1])/2))
        
    #     for j in range(bboxes2.shape[0]):
    #         second_image_center_y = round(bboxes2[j][0] + ((bboxes2[j][2] - bboxes2[j][0])/2))
    #         second_image_center_x = round(bboxes2[j][1] + ((bboxes2[j][3] - bboxes2[j][1])/2))
            
    #         if(abs(first_image_center_y - second_image_center_y) < 60 and abs(first_image_center_x - second_image_center_x) < 20):
                
    #             if(abs(first_image_center_y - second_image_center_y) < 10):
    #                 print("Constant")
    #                 direction_list[j] = "Constant"
    #             elif(first_image_center_y >= second_image_center_y):
    #                 print("yukari")
    #                 direction_list[j] = "yukari"
    #             elif(first_image_center_y < second_image_center_y):
    #                 print("asagi")
    #                 direction_list[j] = "asagi"            
    
    #         else:
    #             print("Irrelevant")
    #             if(direction_list[j] == ""):
    #                 direction_list[j] = "Irrelevant"
    
    # visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = direction_list)
    
    #%%
    # #bbox yönü gibi gibi
    # direction_list = []
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
    #             test = "Constant"
    #             print("Constant")    
    #     direction_list.append(test)
    #visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'], captions = direction_list)
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
    # direction_list = []
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
            
    #     direction_list.append(test)
       
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], captions = direction_list)
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