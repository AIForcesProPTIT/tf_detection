import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers as KL
from tensorflow.keras.applications.resnet import preprocess_input
_RESNET_MODEL_OUTPUT_LAYERS = {
    'resnet_v1_50': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_101': ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out'],
    'resnet_v1_152': ['conv2_block3_out', 'conv3_block8_out',
                      'conv4_block36_out', 'conv5_block3_out'],
}

def get_resnet(image_inputs, name = 'resnet_v1_50'):

    assert name in _RESNET_MODEL_OUTPUT_LAYERS.keys()
    if name == 'resnet_v1_50':
        backbone = keras.applications.ResNet50(
            include_top=False, input_tensor=image_inputs
        )
    elif name == 'resnet_v1_101':
        backbone = keras.applications.ResNet101(
            include_top=False, input_tensor=image_inputs
        )
    elif name == 'resnet_v1_152':
        backbone = keras.applications.ResNet152(
            include_top=False, input_tensor=image_inputs
        )
    outputs = [
        backbone.get_layer(layer_name).output
        for layer_name in _RESNET_MODEL_OUTPUT_LAYERS[name]
    ]
    return keras.Model(
        inputs=[image_inputs], outputs=outputs
    )

