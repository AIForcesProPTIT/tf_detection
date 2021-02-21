import tensorflow as tf
from tensorflow import keras



import numpy  as np
from src.common.box_coder import BoxCoder
from src.common.matcher import Matcher
from src.common import box_utils as ops
def _sum(x) :
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res
class FocalLossAndArgmaxMatcher(tf.losses.Loss):
    def __init__(self, box_coder =None, matcher = None,delta=0.1, alpha=0.25, gamma=1.5,label_smoothing=0):
        super(FocalLossAndArgmaxMatcher, self).__init__(
            reduction="none", name="focalloss_and_argmax_matcher"
        )
        if box_coder is None:
            box_coder = BoxCoder(weights=[1.0,1.0,1.0,1.0])
        
        if matcher is None:
            matcher = Matcher(high_threshold=0.6,low_threshold=0.4,allow_low_quality_matches=False)
        
        self.box_coder = box_coder
        self.matcher = matcher
        self.huber = tf.keras.losses.Huber(
            delta, reduction='sum')

        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def compute_matcher_idx(self,box_true, label_true, anchors):
        matched_idxs = []
        N = box_true.shape.as_list()[0]
        for i in range(N):
            b_t = box_true[i,...]
            l_t = label_true[i,...]
            idx_valid = tf.where(l_t>=0)[...,0]            
            if b_t.shape[0]==0:
                matched_idxs.append( tf.ones(shape=(b_p.shape[0],),dtype=tf.int32) * -1)
                continue
            match_quality_matrix =  ops.iou(b_t ,anchors) 
            matched_idxs.append(self.matcher(match_quality_matrix, valid_idx=idx_valid))
        
        return tf.stack(matched_idxs)

    def call_focal(self, y_true,y_pred):
        alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
        gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

        # compute focal loss multipliers before label smoothing, such that it will
        # not blow up the loss.
        pred_prob = tf.sigmoid(y_pred)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = (1.0 - p_t)**gamma

        # apply label smoothing for cross_entropy for each entry.
        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        # compute the final loss and return
        return alpha_factor * modulating_factor * ce
    
    def call(self, y_true,  head):
        box_true, label_true = y_true
        box_pred = head['bbox_regression'] #(-1,4)
        cls_pred = head ['cls_logits'] #(-1,cl)
        anchors = head['anchors'] #(-1,4)

        matched_idxs  = self.compute_matcher_idx(box_true, label_true, anchors)
        losses = []
        clss_losses = []
        N = box_true.shape.as_list()[0]
        for i in range(N):
            anchors_per_image = anchors
            matched_idxs_per_image = matched_idxs[i,...]
            bbox_regression_per_image = box_pred[i,...]
            cls_logits_per_image = cls_pred[i,...]

            foreground_idxs_per_image = tf.where(matched_idxs_per_image >= 0)[...,0]

            num_foreground = tf.size(foreground_idxs_per_image).numpy()

            matched_gt_boxes_per_image =tf.gather(box_true[i,...], tf.gather( matched_idxs_per_image, foreground_idxs_per_image ) )
            if tf.size(matched_gt_boxes_per_image).numpy()==0:
                matched_gt_boxes_per_image =tf.reshape(matched_gt_boxes_per_image, [-1,4])

            bbox_regression_per_image =tf.gather(bbox_regression_per_image,foreground_idxs_per_image)
            anchors_per_image =tf.gather(anchors_per_image,foreground_idxs_per_image)
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            losses.append(
                self.huber(
                    target_regression, bbox_regression_per_image
                )/ max(1, num_foreground)
            )

            gt_classes_target = label_true[i,...]
            tmp_gt_classes = tf.concat([
                tf.stack([-2,-1,]),
                tf.reshape(gt_classes_target,(-1,))
            ], axis=0)
            gather_indices = tf.maximum(matched_idxs_per_image + 2, 0)
            gathered_tensor = tf.gather(tmp_gt_classes,tf.reshape( gather_indices,(-1,)) )

            depth = tf.shape(cls_logits_per_image)[-1]
            
            gt_classes_target = tf.one_hot(
                gathered_tensor.numpy(),depth
            )
            valid_idxs_per_image =tf.where(matched_idxs_per_image>=-1)[...,0]
            clss_losses.append(
                tf.math.reduce_sum(
                    self.call_focal(
                        tf.gather(gt_classes_target,valid_idxs_per_image),
                        tf.gather(cls_logits_per_image,valid_idxs_per_image)
                )
            ) / max(1, num_foreground))

        return _sum(clss_losses) /max(1, len(box_true)), _sum(losses) / max(1,len(box_true))

        


keras.Model