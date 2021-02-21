import tensorflow as tf
from tensorflow import keras


def share_conv_module(features,norm=None, **kwargs ):
    
    if norm is None or norm == False:
        conv_share =  keras.layers.Conv2D(**kwargs)
        return [conv_share( feature) for feature in features]
    else:
        kwargs['use_bias']=False
        activation = kwargs.pop("activation",None)
        conv_share = keras.layers.Conv2D(**kwargs)
        features = [conv_share( feature) for feature in features]
        kwargs.pop("name")
        if isinstance(norm,bool):
            bn_share = keras.layers.BatchNormalization()
        elif isinstance(norm, keras.layers.Layer):
            bn_share = norm
        features = [bn_share( feature) for feature in features]

        if activation : 
            activation_share = keras.layers.Activation(activation=activation)
            return [activation_share( feature) for feature in features]
        return features

def reshape_to_valid_classifier( features, num_classes):
    shape = features.shape.as_list()# [None, h,w,c]
    assert shape[1] * shape[2] * shape[3] % num_classes == 0
    return keras.layers.Reshape(target_shape=(-1, num_classes)) (features)

def reshape_to_valid_reg( features):
    shape = features.shape.as_list()# [None, h,w,c]
    assert shape[1] * shape[2] * shape[3] % 4 == 0
    return keras.layers.Reshape(target_shape=(-1, 4))(features)