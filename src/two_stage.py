import tensorflow as tf
from tensorflow import keras



from src.backbone.resnet_features import get_resnet
from src.neck.build_neck import build_from_config
from src.head.anchor_based_head import build_anchor_based_head
from src.one_stage import build_head
from src.common.decoder import DecodePredictions
from src.head.roi_head import MultiScaleRoIAlign
def build_backbone( backbone_name = 'resnet_v1_50',
                    image_inputs = (512, 512, 3),
                    **kwargs):
    image_input = keras.layers.Input(shape=image_inputs)
    if backbone_name.startswith("resnet"):
        backbone = get_resnet(image_input, backbone_name) 
    
    return backbone


def build_neck(features ,config_neck:dict, return_config=False):
    # if isinstance(features)
    pad=max(0, 5-len(features))
    neck,config = build_from_config([0,] * pad + features,config = config_neck)
    if  return_config : return neck,config
    return neck

def build_first_head(*args, **kwargs):
    return build_head(*args, **kwargs)

def build_crop_first_head(first_stage_outputs, anchors, features_maps, *args, **kwargs):
    decoder_rpn = DecodePredictions(anchors=anchors,
                                    image_shape=kwargs.get("image_shape",None),
                                    name="fillter_rpn")([first_stage_outputs[0], first_stage_outputs[1]])
    
    head_crop_rpn = MultiScaleRoIAlign(name="multiScale")((decoder_rpn[0],feature_map))

def build_flatten_crop(head_crop_rpn, representation_size = 1024, num_classes = 14, stacked_fc = 2,**kwargs):
    activation = kwargs.get("activation","relu")
    norm = kwargs.get("norm",False)

    head_flatten = keras.layers.Flatten(name="flatten_features")(head_crop_rpn)

    if norm is False:
        for i in range(stacked_fc):
            head_flatten = keras.layers.Dense(representation_size, activation=activation)(head_flatten)
    else:
        for i in range(stacked_fc):
            head_flatten = keras.layers.Dense(representation_size, activation=None, use_bias=False)(head_flatten)
            head_flatten = keras.layers.BatchNormalization()(head_flatten)
            head_flatten = keras.layers.Activation(activation)(head_flatten)

    convs = keras.layers.Dense(num_classes, activation='sigmoid')(head_flatten)
    regs = keras.layers.Dense(num_classes * 4, activation=None)

    return convs,regs






