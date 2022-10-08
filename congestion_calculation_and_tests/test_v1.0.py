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
    DETECTION_MIN_CONFIDENCE = 0.95

    
config = CustomConfig()

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#Loading the model in the inference mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
# loading the trained weights o the custom dataset
model.load_weights("C:\\Users\\MONSTER\\Desktop\\transfer\\Mask_RCNN-final\\logs\\highway20200516T2319\\mask_rcnn_highway_0050.h5", by_name=True)
#img = load_img("C:\\Users\\MONSTER\\Desktop\\transfer\\Mask_RCNN\\test.jpg")
#img = load_img("C:\\Users\\MONSTER\\Desktop\\SonVeriseti\\SonVeriseti\\train\\Vehicle\\vehicle0157.png")
#img = img_to_array(img)

#%%
# class_names = ['background', 'vehicle']

# img = load_img("C:\\Users\\MONSTER\\Desktop\\x.jpg")
# img = img_to_array(img)

# # Run detection
# results = model.detect([img], verbose=1)
 
# # Visualize results
# r = results[0]
# # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
# bboxes = results[0]['rois']

# img2 = load_img("C:\\Users\\MONSTER\\Desktop\\2b.png")
# img2 = img_to_array(img2)

# # Run detection
# results2 = model.detect([img2], verbose=1)
  
# # Visualize results
# r2 = results2[0]
# # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
# bboxes2 = results2[0]['rois']

# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
# visualize.display_instances(img2, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'])

#%%
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
img = load_img("C:\\Users\\MONSTER\\Desktop\\test1.png")
img = img_to_array(img)

# Run detection
results = model.detect([img], verbose=1)
r = results[0]

class_names = ['background', 'vehicle']

# bboxes = results[0]['rois']
# test = ""
# test_array = []
# for b in bboxes:
#     if(abs((b[2] - b[0]) - (b[3] - b[1])) < (((b[2] - b[0]) + (b[3] - b[1]))*2)/100*14):
#         print("araba çarpraz duruyor")    
#         test = "çapraz"
#     elif(b[2] - b[0] > b[3] - b[1]):
#         print("araba dikey duruyor")
#         test = "dikey"
#     else:
#         print("araba yatay duruyor")    
#         test = "yatay"
        
#     test_array.append(test)
   
# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], captions = test_array)
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
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