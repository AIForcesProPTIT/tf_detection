import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import cv2
import albumentations as A

import os,sys,glob
os.environ['test_case'] = 'false'

from src.common.box_coder import *
from src.one_stage import build_model

from src.one_stage import build_model,build_backbone,build_head,build_neck
from src.anchor.anchor_base import AnchorGenerator
from src.data.coco import DataSet

# from src.losses.focaloss_and_argmax_matcher import FocalLossAndArgmaxMatcher

from src.architectures.retina_net import get_default_retina



# build_anchors 

anchor_sizes = tuple( (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)) ) for x in [32, 64, 128, 256, 512])
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
images = tf.zeros(shape=(1,1024,1024,3), dtype=tf.float32)
image_shape = images.shape.as_list()
feature_map_shapes = [
    (image_shape[1] // 2**i, image_shape[2] // 2**i,3) for i in range(3,8)
]

anchors = anchor_generator(images, feature_map_shapes)



# define loss
def bbox_loss(y_true, y_pred, delta = 0.5):
    """y_true: shape = (batch, n_anchors, 4)
       y_pred : shape = (batch, n_anchors, 4)
    """ 
    out = tf.keras.losses.Huber(
            delta, reduction='none')(y_true, y_pred)
    return out
def cls_loss(y_true, y_pred, alpha = 0.25, gamma = 1.5, label_smoothing = 0.):
    """y_true: shape = (batch, n_anchors, 1)
       y_pred : shape = (batch, n_anchors, num_class)
    """ 
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma
    
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    
    return alpha_factor * modulating_factor * ce

def loss_fn(y_true, y_pred):
    matched_gt_boxes,matched_gt_labels, mask_bboxes,mask_labels = y_true
    cls_pred, reg_pred = y_pred
    num_positive = tf.reduce_sum(mask_bboxes, axis=1) # batch,
    
    num_positive = tf.where(num_positive >=0, num_positive, 1.) #batch
    
    reg_loss = bbox_loss(matched_gt_boxes, reg_pred) * mask_bboxes # 
    
    matched_gt_labels = tf.one_hot(matched_gt_labels,  depth=cls_pred.shape.as_list()[-1])
    
    cls_losses = tf.math.reduce_sum(cls_loss(matched_gt_labels, cls_pred),-1) * mask_labels # batch,num_anchors
    
    cls_losses = tf.math.reduce_sum(cls_losses,-1) / num_positive
    
    
    
    return tf.reduce_mean(tf.reduce_sum(reg_loss,axis=1) / num_positive),tf.reduce_mean(cls_losses)




# model train wrapper
class ModelTrainWraper(keras.Model):
    
    def __init__(self, *args, anchor_generator = None,**kwargs):
        super().__init__(*args,**kwargs)
        self.anchors = anchors
    def train_step(self, data):
        images, matched_gt_boxes, matched_gt_labels,\
        mask_bboxes,mask_labels= data
        y_true = (matched_gt_boxes, matched_gt_labels,\
                    mask_bboxes,mask_labels)
        with tf.GradientTape() as tape:
            
            convs, regs = self(images, training=True)
            image_shape = images.shape.as_list()
            loss = self.loss(
                  y_true, (convs, regs))
#             print(loss)
            total_loss = loss[0] * 50. + loss[1] 
            trainable_vars = self.trainable_variables
            
            scaled_loss = total_loss
            optimizer = self.optimizer
            
#         learning_rate = float(tf.keras.backend.get_value(self.optimizer.lr))
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = scaled_gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.compiled_metrics.update_state(y_true, (convs,regs))
        return {'loss_cls':loss[0], 'loss_reg':loss[1], 'total_loss':total_loss}



#--------------------------------------------------Models--------------------------------------------------

# default example
# model = get_default_retina()

# or build model by hand 

backbone_config = {
    'name':'resnet_v1_50',
    'inputs_shape':(1024, 1024, 3)
}

neck_config = {
    'name':'fpn_network',
    'build_node': 
        [
            ('c5_branch_2' , [ {'4' : 'through' , 'kwargs':{} },{'type':'identity'}]),
            ('c4_branch_2' , [ {'3' : 'through'}, {'5' : 'up_sample'},{'type':'add'}]),
            ('c3_branch_2' , [ {'2' : 'through'}, {'6': 'up_sample'}, {'type':'add'}]),
            ('c2_branch_2' , [ {'1' : 'through'}, {'7': 'up_sample'},{'type':'add'}]),
            ('c6_branch_1' , [ {'4' : 'down_sample'},{'type':'identity'}] ),
            ('c7_branch_1' , [ {'9' : 'down_sample'},{'type':'identity'}] ),
            ('return_node' , [7, 6, 5, 9, 10] )
        ]
    # Node will build likes : 
    #"""
        #c6  9  ->  c7   10
        # |
        #c5  4   -> c5'   5
        #            |
        #c4  3   ->  c4'  6
        #            |
        #c3  2   ->  c3'  7
        #            |
        #c2  1   ->  c2'  8

        #c1  0 
        #return [c3:7,c4:6,c5:5,c6:9,c7:10]  = [7, 6, 5, 9, 10]
    #"""
    # some config inside kwargs : build convsModule : example kwargs :{"filters":512, "activation":'relu',"norm":True || your custom norm..} ...
    # todos : make convsModule more stable with SeperateConvs, custom Norm, custom activations ...
    # todos : add more examples for build_node with BiFPN, PA-FPN,... 
}

head_config = {
    'head_name':'anchor_based_head',
    "num_classes":14,
    "num_anchors":9,
    "feat_channels":256,
    "stacked_convs":4
}

inputs,outputs = build_model(
    backbone_config,
    neck_config,
    head_config
)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model = ModelTrainWraper(anchor_generator = anchors, inputs=inputs,outputs=outputs)
model.compile(
    optimizer=optimizer,loss=loss_fn
)


# make short data
import albumentations as A
def get_train_transforms():
    return A.Compose(
        [
        ## RandomSizedCrop not working for some reason. I'll post a thread for this issue soon.
        ## Any help or suggestions are appreciated.
#         A.RandomSizedCrop(min_max_height=(300, 512), height=512, width=512, p=0.5),
#         A.RandomSizedCrop(min_max_height=(300, 1000), height=1000, width=1000, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.9),
        ],p=0.9),
        A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
            ],p=0.1),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Transpose(p=0.2),
        A.Resize(128, 128)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

data = DataSet(img_dir="./data_set/vin_data_dowscale_3x_with_coco/", tranforms=get_train_transforms(),
                anchor_default=anchors)

data_train = data.data_tensor.map(lambda x:(x['img'],x['matched_gt_boxes'],x['matched_gt_labels'],
                                            x['mask_bboxes'],x['mask_labels']))
data_train = data_train.batch(4).prefetch(2)

model.fit( data_train, epochs=10)

model.save_weights("./checkpoints/retinanet.h5")
# model.loss




