


import tensorflow as tf
from tensorflow import keras
import functools

from collections import OrderedDict



def build_through_conv(previous_layer, apply_batch_norm = True, **kwargs):
    if apply_batch_norm :
        kwargs['use_bias']  = False
        activation = kwargs.pop("activation", None)
        conv = keras.layers.Conv2D(**kwargs)(previous_layer)
        kwargs.pop("name",None)
        bn = keras.layers.BatchNormalization()(conv)
        if activation:
            return keras.layers.Activation(activation)(bn)
        return bn


def _pool2d( inputs, height, width, target_height, target_width, pooling_type='max', name=None):
        """Pool the inputs to target height and width."""
        height_stride_size = int((height - 1) // target_height + 1)
        width_stride_size = int((width - 1) // target_width + 1)
        if pooling_type == 'max':
            return tf.keras.layers.MaxPooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',name=name
                )(inputs)
        if pooling_type == 'avg':
            return tf.keras.layers.AveragePooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',name=name,
                )(inputs)
        raise ValueError('Unsupported pooling type {}.'.format(pooling_type))

def build_down_sample(previous_layer, apply_batch_norm = True , **kwargs):
    
    height,width = previous_layer.shape.as_list()[-3:-1]  #bhwc
    
    target_height, target_width = (height + 1) // 2, (width + 1) // 2

    pool = _pool2d( previous_layer, height, width,
                    target_height, target_width,
                    pooling_type=kwargs.pop('pooling_type','max'),
                    name=kwargs.pop("name",None))
    
    return build_through_conv(pool,apply_batch_norm=apply_batch_norm, **kwargs)


def build_up_sample(previous_layer , apply_batch_norm = True , **kwargs):
    interpolation=kwargs.pop('interpolation','nearest') 
    name = kwargs.pop("name",None)
    conv_1_1 = build_through_conv(previous_layer,apply_batch_norm=apply_batch_norm, **kwargs)
    up_sample = keras.layers.UpSampling2D(size=(2,2),interpolation=interpolation, name=name)(conv_1_1)
    return up_sample