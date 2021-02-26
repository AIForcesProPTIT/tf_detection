
import tensorflow as tf
from tensorflow import keras


def bbox_loss(y_true, y_pred, delta = 0.5):
    """y_true: shape = (batch, n_anchors, 4)
       y_pred : shape = (batch, n_anchors, 4)
    """ 
    out = tf.keras.losses.Huber(
            delta, reduction='none')(y_true, y_pred)
    return out
    
def cls_loss(y_true, y_pred, alpha = 0.25, gamma = 2., label_smoothing = 0.):
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

def loss_fn(y_true, y_pred, alpha = 0.25, gamma = 2., label_smoothing = 0.):
    
    matched_gt_boxes,matched_gt_labels, mask_bboxes,mask_labels = y_true
    cls_pred, reg_pred = y_pred

    num_positive = tf.reduce_sum(mask_bboxes, axis=1) # batch,
    
    num_positive = tf.where(num_positive > 0, num_positive, 1.) #batch
    
    reg_loss = bbox_loss(matched_gt_boxes, reg_pred) * mask_bboxes # 
    
    matched_gt_labels = tf.one_hot(matched_gt_labels,  depth=cls_pred.shape.as_list()[-1])
    
    cls_losses = tf.math.reduce_sum(cls_loss(matched_gt_labels, cls_pred,
                                    alpha = alpha, gamma = gamma, label_smoothing = 0.),-1) * mask_labels # batch,num_anchors
    
    cls_losses = tf.math.reduce_sum(cls_losses,-1) / num_positive
    
    return tf.reduce_mean(tf.reduce_sum(reg_loss,axis=1) / num_positive),tf.reduce_mean(cls_losses)


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss
def head_loss_func(y_true, y_pred, num_classes = 14, softmax_=False):
    
    softmax_valid = int(softmax_)
    (target_regressiones, # None,200,4
    target_clasifications, # None,200
    mask_regressiones,     # None,200
    mask_clasifications) = y_true # None,200
    
    convs, regs = y_pred # None,200,14 - None,200,14 * 4
    
    # sample_regs
    regs = tf.reshape(regs, (-1, num_classes, 4)) # None*200,14,4
    mask_clasifications = tf.reshape(mask_clasifications, (-1,))
    target_regressiones = tf.reshape(target_regressiones, (-1,4))
    target_clasifications = tf.reshape(target_clasifications, (-1,))
    mask_regressiones = tf.reshape(mask_regressiones,(-1,))
    convs = tf.reshape(convs, (-1, num_classes))
    roi_valid =tf.cast(tf.where(mask_clasifications >= softmax_valid )[...,0],dtype=tf.int32) 
    
    
    positive_roi_class_ids =tf.cast(tf.gather(mask_clasifications, roi_valid),dtype=tf.int32)
    indices = tf.stack([roi_valid, positive_roi_class_ids], axis=1)
    
     # None,valid_num
    
#     print(target_regressiones.shape.as_list())
    pred_bbox = tf.gather_nd(regs, indices) # indices :shape = 2,regs rank = 3 -> gather_nd
#     print(pred_bbox.shape.as_list())
    loss_reg =tf.keras.losses.Huber(
                0.5, reduction='none')(target_regressiones, pred_bbox)
#     print(loss_reg)
    loss_reg = loss_reg * mask_regressiones[...,None]
#     print(mask_regressiones)
#     print(target_regressiones)
#     print(pred_bbox)
   
    #-------------------------------loss-cls--------------------------#
    
    
    if not softmax_:
        matched_gt_labels = tf.one_hot(target_clasifications,  depth=convs.shape.as_list()[-1])
        loss_cls = tf.math.reduce_sum(
                            cls_loss(matched_gt_labels,convs ), -1
                    ) 
        loss_cls = loss_cls * mask_clasifications
        
        loss_cls = tf.math.reduce_sum(loss_cls) / tf.maximum(tf.math.reduce_sum(mask_clasifications),1)
        
        return tf.math.reduce_sum(loss_reg) / tf.maximum(tf.math.reduce_sum(mask_regressiones),1), loss_cls

    # implement softmax_here


def rpn_loss_func(y_true, y_pred):
    (matched_gt_boxes, matched_gt_labels,\
                    mask_bboxes,mask_labels) = y_true
    
    (rpn_convs, rpn_regs) = y_pred  # None,num_anchor,1 - None,num_anchor,4
    
    label_logits = tf.cast(tf.where(matched_gt_labels >= 0, 1, 0),tf.int32)
#     print(label_logits)
    return loss_fn(
        (matched_gt_boxes, label_logits,\
                    mask_bboxes,mask_labels),
        (rpn_convs, rpn_regs)
    )