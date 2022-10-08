import os
import cv2

#list of image file names, you can read them however you want.
arr = os.listdir("C:\\Users\\MONSTER\\Desktop\\SonVeriseti\\SonVeriseti\\train\\Vehicle")
arr.remove("desktop.ini")
arr.remove("via_region_data.json")
# path of image folder + \\
path = "C:\\Users\\MONSTER\\Desktop\\SonVeriseti\\SonVeriseti\\train\\Vehicle\\"

#image height, width list
image_heights = []
image_widths = []

#read image height, width
for i in range(len(arr)):
    im = cv2.imread(path + arr[i])
    h, w, c = im.shape
    image_widths.append(w)
    image_heights.append(h)



#%%

import json
data = {}

#info tag
data['info'] = {'description': 'Test Dataset', 'url': 'https://www.linkedin.com/in/burlakorkmaz/', 'version': '0.1', 'year': 2020, 'contributor': 'Burla Nur Korkmaz', 'date_created': '2020/04/21'}

#licenses tag
data['licenses'] = []
data['licenses'].append({'url': 'https://dar.vin/OWGjN', 'id': 0, 'name': 'You can use it'})

#images tag
data['images'] = []
for i in range(len(arr)):
    file_name = arr[i]
    width = image_widths[i]
    height = image_heights[i]
    id = i
    data['images'].append({'license': 0, 'file_name': file_name, 'width': width, 'height': height, 'id': id})



    
#%%

#calculate polygon area from list
def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

import itertools
def mygrouper(n, iterable):
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.zip_longest(*args))

#load mask r cnn annotations
with open("C:\\Users\\MONSTER\\Desktop\\SonVeriseti\\SonVeriseti\\train\\Vehicle\\via_region_data.json") as f:
    annotations = json.load(f)

data['annotations'] = []

image_ids = []
category_ids = []
ids = []
all_points = []
id_counter = 0
#crowd_list = []

#all annotations
for i in range(len(annotations)):    
    #take every regions in all annotations
    regions = annotations[list(annotations.keys())[i]]['regions']
    for j in range(len(regions)):
        
        #x points
        x_points = annotations[list(annotations.keys())[i]]['regions'][str(j)]['shape_attributes']['all_points_x']
        #y points
        y_points = annotations[list(annotations.keys())[i]]['regions'][str(j)]['shape_attributes']['all_points_y']
        #merge them for coco format
        points = [j for i in zip(x_points, y_points) for j in i]
        #list of all points
        all_points.append(points)
        image_ids.append(i)
        
        # category_holder = annotations[list(annotations.keys())[i]]['regions'][str(j)]['region_attributes']['name']
        # if (category_holder == 'highway'):
        #     category_ids.append(1)
        # else if (category_holder == 'vehicle'):
        #     category_ids.append(2)
        
        #append id of categories. if there is more than 1, check for name of them, then append the values. It must be the same with categories tag
        category_ids.append(1)
        ids.append(id_counter)

        #bbox
        x_min = min(annotations[list(annotations.keys())[i]]['regions'][str(j)]['shape_attributes']['all_points_x'])
        x_max = max(annotations[list(annotations.keys())[i]]['regions'][str(j)]['shape_attributes']['all_points_x'])
        y_min = min(annotations[list(annotations.keys())[i]]['regions'][str(j)]['shape_attributes']['all_points_y'])
        y_max = max(annotations[list(annotations.keys())[i]]['regions'][str(j)]['shape_attributes']['all_points_y'])

        #area
        polygon_area = all_points[id_counter]
        polygon_area = list(mygrouper(2, polygon_area))
        polygon_area = PolygonArea(polygon_area)
        
        #write to json. create an iscrowd list then append its values in this loop if you need multiple values.
        data['annotations'].append({'segmentation': [all_points[id_counter]], 'iscrowd': 0, 'image_id': image_ids[id_counter], 'category_id': category_ids[id_counter], 'id': ids[id_counter], 'bbox': [x_min, y_min, (x_max - x_min), (y_max - y_min)], 'area': polygon_area})
        
        #region counter
        id_counter +=1
        
#%%
        
#categories tag. add more if you have more than 1 category
data['categories'] = []
data['categories'].append({'supercategory': 'road', 'id': 1, 'name': 'vehicle'})
#data['categories'].append({'supercategory': 'fun', 'id': 2, 'name': 'balloon'})

#%%

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)