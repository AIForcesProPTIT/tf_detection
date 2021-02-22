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

class AssignTargetTF(object):
    def __init__(self ,box_coder = None,matcher = None, **kwargs):
        if box_coder is None:
            box_coder = BoxCoder(weights=[10.,10.,5.,5.])
        if matcher is None:
            matcher = Matcher(0.6, 0.4,allow_low_quality_matches=False)
        self.box_coder = box_coder
        self.matcher = matcher
        self.__dict__.update(**kwargs)
    
    def __call__(self,
                bboxes:tf.TensorArray,
                labels:tf.TensorArray,
                proposal:tf.TensorArray):
        """anchors: (N,4)
           bboxes : (batch_size, M, 4)
           labels : (batch_size , M) # -1 for idx
           proposal: (batch_size, numboxes, (x1, y1, x2, y2))
        """
        dtype = proposal.dtype
        bboxes = tf.cast(bboxes, dtype=dtype)

        list_bboxes = tf.unstack(bboxes, axis=0)
        list_proposal = tf.unstack(proposal, axis=0)
        list_labels = tf.unstack(labels, axis=0)

        mask_regressiones = []
        mask_clasifications = []
        target_regressiones = []
        target_clasifications = []

        for bboxes_un, labels_un,proposal_un in zip(list_bboxes, list_labels, list_proposal):
            # shape (M,4) , (N,4)
            matrix_iou = iou(bboxes_un, proposal_un)
            # map idx prosal to bboxes_un
            matched_idx = self.matcher(matrix_iou)  # idx shape = (N,)
            


            fake_bboxes = tf.concat([
                tf.convert_to_tensor(np.array([0.,] *8).reshape(2,4), dtype=dtype),
                tf.reshape(bboxes_un,(-1,4))
            ], axis=0)

            fake_labels = tf.concat([
                tf.convert_to_tensor(np.array([-2,-1]).reshape(-1,1)),
                tf.reshape(labels_un,(-1,1))
            ], axis=0)

            matched_gt_boxes = tf.gather(fake_bboxes,matched_idx + 2)

            matched_gt_labels = tf.gather(fake_labels, matched_idx + 2)

            target_regression = self.box_coder.encode_single(matched_gt_boxes, proposal_un)

            
            target_regressiones.append(target_regression)
            target_clasifications.append(matched_gt_labels)
            
            mask_regressiones.append(tf.where(matched_idx >=0, 1., 0.))
            mask_clasifications.append(tf.where(matched_idx >= -1, 1., 0.))
        
        target_regressiones = tf.stack(target_regressiones,axis=0)
        target_clasifications = tf.stack(target_clasifications, axis=0)
        mask_regressiones = tf.stack(mask_regressiones, axis=0)
        mask_clasifications = tf.stack(mask_clasifications, axis=0)

        return target_regressiones,target_clasifications,mask_regressiones,mask_clasifications


        
        

