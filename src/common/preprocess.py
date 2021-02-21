import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      ratio: The scaling factor used to resize the image
    """
    image_shape = np.array(image.shape[:2]).astype(np.float)

    if jitter is not None:
        min_side = np.random.uniform( jitter[0], jitter[1],size=())
    ratio = min_side / np.min(image_shape)
    if ratio * np.max(image_shape) > max_side:
        ratio = max_side / np.max(image_shape)
    image_shape = ratio * image_shape
    image_shape = image_shape.astype(np.int32)
    image = cv2.resize(image, (image_shape[1],image_shape[0]))

    
    padded_image_shape = np.ceil(
        image_shape/stride
    ) * stride
    
    padded_image_shape = (int(padded_image_shape[0]),int(padded_image_shape[1]))
    padded_image = np.zeros(shape=padded_image_shape+(3,))
    padded_image[0:image.shape[0],0:image.shape[1],:] = image
    
    return padded_image,ratio
