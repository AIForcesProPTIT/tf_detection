import tensorflow as tf
from tensorflow import keras

from src.head.common import share_conv_module, reshape_to_valid_classifier, reshape_to_valid_reg
import logging
logger = logging.getLogger(__name__)

# todos : add example rpn_head : num_classes = 1,num_anchors = 0, feat_chanels = 0, stacked_convs = 0
def build_anchor_based_head(  features,
                        num_classes=1,
                        num_anchors = 9,
                        feat_channels=256,
                        stacked_convs=4,
                        **kwargs_conv):
    
    # print(features)
    if len(set([i.shape.as_list()[-1] for i in features]))!=1:
        logger.warning("head recevied different shape so add one_stack to necks")
        for i in range(len(features)):
            features[i] = keras.layers.Conv2D( feat_channels, 1, strides=(1,1),
                                         padding='SAME', activation=None,use_bias=False)( features[i] )
            
            norm = kwargs_conv.get("norm",None)
            if norm is None or norm == False:
                features[i] = keras.layers.Activation(kwargs_conv.get("activation","relu"))( features[i] )

            elif isinstance(norm,bool):
                features[i] = keras.layers.BatchNormalization()( features[i] )
            else:
                features[i] = norm(features[i])
    cls_convs = None
    reg_convs = None
    for i in range(stacked_convs):
        name_cls = f'stacked_convs_cls_{i}'
        name_reg = f'stacked_convs_reg_{i}'
        if cls_convs is None:
            kwargs = dict(
                filters = feat_channels,
                kernel_size= 3,
                strides = 1,
                padding='SAME',
                activation='swish',
            )
            for item in kwargs_conv.keys():
                if item in kwargs.keys():
                    kwargs[item] = kwargs_conv[item]
            cls_convs = share_conv_module( features, norm=kwargs_conv.pop("norm",True), **kwargs,name=name_cls)
            reg_convs = share_conv_module( features, norm=kwargs_conv.pop("norm",True), **kwargs, name=name_reg)
        else:
            kwargs = dict(
                filters = feat_channels,
                kernel_size= 3,
                strides = 1,
                padding='SAME',
                activation='swish'
                
            )
            for item in kwargs_conv.keys():
                if item in kwargs.keys():
                    kwargs[item] = kwargs_conv[item]
            cls_convs = share_conv_module(cls_convs, norm=kwargs_conv.pop("norm",True), **kwargs,name=name_cls)
            reg_convs = share_conv_module(reg_convs, norm=kwargs_conv.pop("norm",True), **kwargs, name=name_reg) 
    
    if cls_convs is None:
        cls_convs, reg_convs = features, features
    
    head_cl_conv_share = keras.layers.Conv2D(
            num_anchors * num_classes,
            3,
            strides=1,
            padding='SAME'
        )
    head_reg_conv_share = keras.layers.Conv2D(
            num_anchors * 4,
            3,
            strides=1,
            padding='SAME'
        )

    
    out_convs = [reshape_to_valid_classifier(head_cl_conv_share(i), num_classes) for i in cls_convs]
    out_regs = [ reshape_to_valid_reg(head_reg_conv_share(i)) for i in reg_convs]
   
    return keras.layers.Concatenate(axis=1, name='head_classifier')(out_convs),\
           keras.layers.Concatenate(axis=1, name='head_regression')(out_regs)
            

    
