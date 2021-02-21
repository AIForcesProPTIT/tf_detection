from tensorflow import keras
import numpy as np
import tensorflow as tf

from src.common.box_coder import BoxCoder,BoxCoderNumpy
from src.common.matcher import Matcher, MatcherNumpy
from src.common.box_utils import iou


class AssignTarget(object):
    def __init__(self , box_coder = BoxCoderNumpy(weights=[10.,10.,5.,5.]),matcher =None, **kwargs):
        self.box_coder = box_coder
        if matcher is None:
            matcher = MatcherNumpy(high_threshold=0.6, low_threshold=0.4, allow_low_quality_matches=False)
        self.matcher = matcher
        self.__dict__.update(**kwargs)
    
    def __call__(self, anchors, bboxes, labels):
        anchors = tf.convert_to_tensor(anchors, dtype= tf.float32)
        bboxes = tf.convert_to_tensor(bboxes, dtype= tf.float32)
        matrix_iou = iou(bboxes, anchors) 
        matrix_iou = matrix_iou.numpy()
        matched_idx =  self.matcher(matrix_iou) # N
        fake_bboxes = np.concatenate([np.array([-1.,-1.,-1.,-1.]).reshape(-1,4), np.array([-1., -1., -1., -1.]).reshape(-1,4), bboxes], axis=0)
        fake_labels = np.concatenate([np.array([-2]), np.array([-1]), labels], axis=0)
        matched_gt_boxes = fake_bboxes[matched_idx + 2]
        matched_gt_labels = fake_labels[matched_idx + 2]
        target_regression = self.box_coder.encode_single(matched_gt_boxes, anchors)

        mask_bboxes = np.where(matched_idx >=0 , 1.0, 0.)
        mask_labels = np.where( matched_idx >= -1, 1., 0.)
        return target_regression, matched_gt_labels,mask_bboxes, mask_labels
        
        

