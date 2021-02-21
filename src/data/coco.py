import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import os,sys,glob,json

import matplotlib.pyplot as plt
from functools import partial
import cv2

from src.common.preprocess import resize_and_pad_image
from src.common.box_coder import *
from src.common.visualize import draw_bbox,draw_bbox_small
from src.common.assign_target import AssignTarget
import albumentations as A


class DataSet():
    
    label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]
    def __init__(self, img_dir=os.getcwd(), tranforms =lambda x: x, anchor_default = np.array([]),
                assigner = None ):
        if assigner is None: assigner = AssignTarget()
        self.assigner = assigner
        self.img_dir = img_dir
        self.anchor_default = anchor_default
        with open(os.path.join(img_dir,
                               "train_annotations_2.json"),'r') as f:
            annotations = json.load(f)
        annotations['categories'].sort(key=lambda x:x['id'])
        self.annotations = annotations
        self.tranforms= tranforms
        self.data_tensor =  tf.data.Dataset.from_generator(partial(DataSet.gen_data, 
                                                                   annotations,
                                                                   self.img_dir,
                                                                   self.tranforms,
                                                                   self.anchor_default,
                                                                   self.assigner),
                                                           output_shapes={
                'img':(None,None,3),
                'bboxes':(None,4),
                'labels':(None,),
                'matched_gt_boxes':(None,4),
                'matched_gt_labels':(None,),
                'mask_bboxes':(None,),
                'mask_labels':(None,)
            }, output_types={
                'img':tf.float32,
                'bboxes':tf.float32,
                'labels':tf.int32,
                'matched_gt_boxes':tf.float32,
                'matched_gt_labels':tf.int32,
                'mask_bboxes':tf.float32,
                'mask_labels':tf.float32
                
            })
        
    @staticmethod
    def gen_data(data, img_dir, tranforms, anchor_default,assigner):

        for image_info, annotation in zip(data['images'],data['annotations_group_by_image_id']):
            path = os.path.join(img_dir, image_info['file_name']) # tf.string
            img = cv2.imread(path)
            bboxes = []
            labels = []
            areas = []
            for item in annotation:
                bboxes.append([item['x_min'],item['y_min'],item['x_max'], item['y_max']])
                labels.append(item['category_id'])
                areas.append(item['area'])
            
            sample = tranforms(image=img, bboxes = bboxes, labels = labels)
            sample['image'] = sample['image']  / 127.5 - 1
            data_yield = {
                'img':sample['image'],
                'bboxes':np.array(sample['bboxes'], dtype=np.float).reshape(-1,4),
                'labels':np.array(sample['labels'], dtype=np.int32).reshape(-1,)
            }

            matched_gt_boxes, matched_gt_labels,mask_bboxes, mask_labels = assigner(anchor_default,
                                                                           data_yield['bboxes'], data_yield['labels'])
            data_yield['matched_gt_boxes'] = matched_gt_boxes
            data_yield['matched_gt_labels'] = matched_gt_labels
            data_yield['mask_bboxes']  = mask_bboxes
            data_yield['mask_labels'] = mask_labels
#             print(data_yield)
            yield data_yield
    @staticmethod
    def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    #     print(x_min,x_max,y_min,y_max)
    #     print(img.shape)
        color = tuple(int(i) for i in color)
        print(color,bbox, class_name)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA,
        )
        return img
    
    def visualize_sample(self,sample):
        img = sample['img']
        bboxes = sample ['bboxes']
        labels = sample['labels']

        if hasattr(bboxes,'numpy'):
            bboxes= bboxes.numpy()
            img = img.numpy()
            labels = labels.numpy()

        class_names = [ self.annotations['categories'][i]['name'] for  i in labels]

        for i,(box, class_name) in enumerate(zip(bboxes,class_names)):
            
            img = DataSet.visualize_bbox(img, box, class_name, color=self.label2color[labels[i]])

        return img 
        
        