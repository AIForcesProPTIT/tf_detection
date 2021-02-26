from tensorflow import keras
import numpy as np
import tensorflow as tf

from src.common.box_coder import BoxCoder,BoxCoderNumpy
from src.common.matcher import Matcher, MatcherNumpy
from src.common.box_utils import iou,coordinates_to_center,center_to_coordinates


def compute_iou(boxes1_corners, boxes2_corners):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1 = coordinates_to_center(boxes1_corners)
    boxes2 = coordinates_to_center(boxes2_corners)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

class AssignTarget(object):
    def __init__(self , box_coder = BoxCoderNumpy(weights=[10.,10.,5.,5.]),matcher =None, **kwargs):
        self.box_coder = box_coder
        if matcher is None:
            matcher = MatcherNumpy(high_threshold=0.5, low_threshold=0.4, allow_low_quality_matches=False)
        self.matcher = matcher
        self.__dict__.update(**kwargs)
    
    def __call__(self, anchors, bboxes, labels):
        anchors = tf.convert_to_tensor(anchors, dtype= tf.float32)
        bboxes = tf.convert_to_tensor(bboxes, dtype= tf.float32)
        matrix_iou = compute_iou(bboxes, anchors) 
        matrix_iou = matrix_iou.numpy()
        

        matched_idx =  self.matcher(matrix_iou) # N
        fake_bboxes = np.concatenate([np.array([-1.,-1.,-1.,-1.]).reshape(-1,4), np.array([-1., -1., -1., -1.]).reshape(-1,4), bboxes], axis=0)
        fake_labels = np.concatenate([np.array([-2]), np.array([-1]), labels], axis=0)
        matched_gt_boxes = fake_bboxes[matched_idx + 2]
        matched_gt_labels = fake_labels[matched_idx + 2]
        target_regression = self.box_coder.encode_single(matched_gt_boxes, anchors)

        mask_bboxes = np.where(matched_idx >= 0 , 1.0, 0.)
        mask_labels = np.where( matched_idx >= -1 , 1., 0.)
        return target_regression, matched_gt_labels,mask_bboxes, mask_labels

class AssignTargetTF(object):
    def __init__(self ,box_coder = None,matcher = None, **kwargs):
        if box_coder is None:
            box_coder = BoxCoder(weights=[10.,10.,5.,5.])
        if matcher is None:
            matcher = Matcher(0.5, 0.4,allow_low_quality_matches=False)
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
        proposal.set_shape([len(list_bboxes),] + proposal.shape.as_list()[1:])
        list_proposal = tf.unstack(proposal, axis=0)
        list_labels = tf.unstack(labels, axis=0)

        mask_regressiones = []
        mask_clasifications = []
        target_regressiones = []
        target_clasifications = []

        for bboxes_un, labels_un,proposal_un in zip(list_bboxes, list_labels, list_proposal):
            # shape (M,4) , (N,4)
            # first gather 
            valid_idx = tf.where(labels_un >=0)
            valid_idx = tf.reshape(valid_idx, (-1,))
            bboxes_un = tf.gather(bboxes_un, valid_idx)
            labels_un =  tf.gather(labels_un, valid_idx)
            # assert 
            
            matrix_iou = iou(bboxes_un, proposal_un)
            # print(matrix_iou,tf.shape(matrix_iou), tf.shape(matrix_iou)[0])
            # map idx prosal to bboxes_un
            matched_idx = self.matcher(matrix_iou)  # idx shape = (N,)
            # matched_idx = tf.where(valid_idx == 1, matched_idx, valid_idx )


            fake_bboxes = tf.concat([
                tf.convert_to_tensor(np.array([0.,] *8).reshape(2,4), dtype=dtype),
                tf.reshape(bboxes_un,(-1,4))
            ], axis=0)

            fake_labels = tf.concat([
                tf.convert_to_tensor(np.array([-2,-1]).reshape(-1,1), dtype = labels_un.dtype),
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


        
        

