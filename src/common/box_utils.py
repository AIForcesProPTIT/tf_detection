import tensorflow as tf

from tensorflow import keras

def center_to_coordinates(boxes):
    '''convert box format [x,y,w,h] -> [x1,y1,x2,y2]
        box: tensor with shape [N,4]
    '''
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def coordinates_to_center(boxes):
    '''convert box format [x1,y1,x2,y2] -> [x,y,w,h]
        box: tensor with shape [N,4]
    '''
    boxes =  tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )
    boxes = tf.maximum(boxes, 0.0001)
    return boxes



    

def area(boxlist, scope=None):
  """Computes area of boxes.
  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.
  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope('Area'):
    x_min, y_min, x_max, y_max = tf.split(boxlist,num_or_size_splits=4, axis=1)
    return tf.squeeze((x_max - x_min) * (y_max - y_min), [1])

def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope( 'Intersection'):
    x_min1,y_min1,x_max1,y_max1 = tf.split(boxlist1,num_or_size_splits=4, axis=1)
    x_min2,y_min2,x_max2,y_max2 = tf.split(boxlist2,num_or_size_splits=4, axis=1)
    
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    # print
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def matched_intersection(boxlist1, boxlist2, scope=None):
  """Compute intersection areas between corresponding boxes in two boxlists.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding N boxes
    scope: name scope.
  Returns:
    a tensor with shape [N] representing pairwise intersections
  """
  with tf.name_scope(scope, 'MatchedIntersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    min_ymax = tf.minimum(y_max1, y_max2)
    max_ymin = tf.maximum(y_min1, y_min2)
    intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
    min_xmax = tf.minimum(x_max1, x_max2)
    max_xmin = tf.maximum(x_min1, x_min2)
    intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)
    return tf.reshape(intersect_heights * intersect_widths, [-1])


def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope('IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

import torchvision





