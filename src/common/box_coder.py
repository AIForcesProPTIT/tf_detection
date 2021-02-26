

# import tensorflow as tf

# from tensorflow import keras


import numpy as np
import math
from src.common.box_utils import coordinates_to_center,center_to_coordinates
from .box_utils import coordinates_to_center,center_to_coordinates
import tensorflow as tf
from tensorflow import keras
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    proposals_x1,proposals_y1,proposals_x2,proposals_y2 = tf.unstack(proposals,axis=1)
    reference_boxes_x1,reference_boxes_y1,reference_boxes_x2,reference_boxes_y2 = tf.unstack(reference_boxes, axis=1)
    

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    ex_widths = tf.maximum(ex_widths, 0.1)
    ex_heights = tf.maximum(ex_heights, 0.1)
    gt_widths = reference_boxes_x2 - reference_boxes_x1 
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_widths = tf.maximum(gt_widths, 0.1)
    gt_heights = tf.maximum(gt_heights, 0.1)
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * tf.math.log(gt_widths / ex_widths)
    targets_dh = wh * tf.math.log(gt_heights / ex_heights)
    
    targets = tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(2000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes =tf.concat(reference_boxes, axis=0)  
        proposals = tf.concat(proposals, axis=0)

        targets = self.encode_single(reference_boxes, proposals)

        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        weights =tf.convert_to_tensor(self.weights, dtype=dtype)
        targets = tf.convert_to_tensor(encode_boxes(reference_boxes, proposals, weights))
        return targets

    

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw =tf.clip_by_value(dw,clip_value_min=-self.bbox_xform_clip ,clip_value_max=self.bbox_xform_clip )
        dh =tf.clip_by_value(dh,clip_value_min=-self.bbox_xform_clip ,clip_value_max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w =tf.math.exp(dw) * widths[:, None]
        pred_h =tf.math.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes2 = pred_ctr_y - 0.5 * pred_h
        pred_boxes3 = pred_ctr_x + 0.5 * pred_w
        pred_boxes4 = pred_ctr_y + 0.5 * pred_h
        # print(pred_boxes4.shape)
        pred_boxes = tf.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), axis=2)
        return tf.squeeze(pred_boxes, axis=1)


class BoxCoderNumpy(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(2000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    @staticmethod
    def encode_numpy(reference_boxes, proposals, weights):
        # x,y,x2,y2
        wx = weights[0]
        wy = weights[1]
        ww = weights[2]
        wh = weights[3]
        # print(weights)
        proposals_x1,proposals_y1,proposals_x2,proposals_y2 = [proposals[...,i] for i in range(4)]
        reference_boxes_x1,reference_boxes_y1,reference_boxes_x2,reference_boxes_y2 = [reference_boxes[...,i] for i in range(4)]
        

        # implementation starts here
        ex_widths = proposals_x2 - proposals_x1
        ex_heights = proposals_y2 - proposals_y1
        ex_widths = np.maximum(ex_widths, 0.1)
        ex_heights = np.maximum(ex_heights, 0.1)
        ex_ctr_x = proposals_x1 + 0.5 * ex_widths
        ex_ctr_y = proposals_y1 + 0.5 * ex_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_widths = np.maximum(gt_widths, 0.1)
        gt_heights = np.maximum(gt_heights, 0.1)
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * np.log(gt_widths / ex_widths)
        targets_dh = wh * np.log(gt_heights / ex_heights)
        # print(targets_dh.shape, targets_dw.shape, targets_dx.shape, targets_dy.shape)
        targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
        return targets


    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        # dtype = reference_boxes.dtype
        weights =np.array(self.weights,dtype=np.float)
        targets =BoxCoderNumpy.encode_numpy(reference_boxes, proposals, weights)
        return targets

    

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = np.minimum(dw, self.bbox_xform_clip)
        dh = np.minimum(dh, self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w =np.exp(dw) * widths[:, None]
        pred_h =np.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes2 = pred_ctr_y - 0.5 * pred_h
        pred_boxes3 = pred_ctr_x + 0.5 * pred_w
        pred_boxes4 = pred_ctr_y + 0.5 * pred_h
        # print(pred_boxes4.shape)
        pred_boxes = np.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), axis=2)
        return np.squeeze(pred_boxes, axis=1)

def test():
    import os
    # previous_os = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    import numpy as np

    box_ref = np.array([
        0.,0.,120.,120.
    ]).reshape(-1,4)

    box_proposal = np.array([
        0.,0.,60.,60.
    ]).reshape(-1,4) # anchors

    box_coder =BoxCoderNumpy(weights=[1.0,1.0,1.0,1.0])

    out_encode =  box_coder.encode_single(box_ref, box_proposal)

    print(out_encode)

    out_decode = box_coder.decode_single(out_encode, box_proposal)
    print(out_decode)
import os
if os.environ.get('test_case','false') == 'true':
    test()


