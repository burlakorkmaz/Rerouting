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
    DETECTION_MIN_CONFIDENCE = 0.7

    
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
img = load_img("C:\\Users\\arasu\\Desktop\\otherLaptop\\1f.jpg")
img = img_to_array(img)

# Run detection
results = model.detect([img], verbose=1)

class_names = ['background', 'vehicle']
   
# Visualize results
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


img = load_img("C:\\Users\\arasu\\Desktop\\otherLaptop\\1f.jpg")
img = img_to_array(img)

# imgwhite = load_img("C:\\Users\\MONSTER\\Desktop\\a-white.png")
# imgwhite = img_to_array(imgwhite)

# Run detection
results = model.detect([img], verbose=1)
r = results[0]

class_names = ['background', 'vehicle']

bboxes = results[0]['rois']
test = ""
test_array = []
for b in bboxes:
    if(abs((b[2] - b[0]) - (b[3] - b[1])) < 100):
        print("araba çarpraz duruyor")    
        # test = "Diagonal"
    elif(b[2] - b[0] > b[3] - b[1]):
        print("araba dikey duruyor")
        # test = "Vertical"
    else:
        print("araba yatay duruyor")    
        # test = "Horizontal"
        
    test_array.append(test)
   
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], captions = test_array)
# visualize.display_instances(imgwhite, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], captions = test_array)

#%%
a = model.keras_model.summary()






#%%
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