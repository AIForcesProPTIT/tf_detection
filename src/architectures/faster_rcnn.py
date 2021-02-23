from src.two_stage import build_model


import tensorflow as tf
from tensorflow import keras
from src.anchor.anchor_base import AnchorGenerator

def get_faster_rcnn_fpn_50(
                           image_shape = (512,512,3),
                           num_classes = 14,
                           aspect_ratios=None,
                           anchor_sizes = None):
    
    anchor_generator = None
    if aspect_ratios is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) # c2,c3,c4,c5,c6
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    # build_anchors

    images_fake = tf.random.normal(shape=(1,) + image_shape)

    
    feature_map_shapes = [
        (image_shape[0] // 2**i, image_shape[1] // 2**i,3) for i in range(2,7)
    ]
    
    anchors = anchor_generator(images_fake, feature_map_shapes)

    backbone_config = dict(
        name = 'resnet_v1_50',
        image_inputs = image_shape
    )
    neck_config = dict(
        name = "fpn_use_c5_max",
        build_node = 
            [
                ('c5_branch_2' , [ {'4' : 'through' , 'kwargs':{'activation':'linear'} },{'type':'identity'}]),
                ('c4_branch_2' , [ {'3' : 'through','kwargs':{'activation':'linear'}}, {'5' : 'up_sample'},{'type':'add'}]),
                ('c3_branch_2' , [ {'2' : 'through','kwargs':{'activation':'linear'}}, {'6': 'up_sample'}, {'type':'add'}]),
                ('c2_branch_2' , [ {'1' : 'through','kwargs':{'activation':'linear'}}, {'7': 'up_sample'},{'type':'add'}]),
                ('c6_branch_1' , [ {'4' : 'down_sample'},{'type':'identity'}] ),
                ('return_node' , [8,7, 6, 5, 9])
            ]
    )

    rpn_config = dict(
        name_head = 'anchor_based_head',
        num_classes=1,
        num_anchors= anchor_generator.num_anchors_per_location()[0], # simple 
        feat_channels=0,
        stacked_convs=0,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=200,
        max_detections=200,
        max_detections_per_class_training=200,
        max_detections_training=200,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        anchors = anchors,

    )

    # multi scale rois
    roi_config = dict(

        crop_size=(7, 7),
        canonical_size = 56.,
        min_level=0, max_level=4,
        method ='bilinear',
        image_shape=image_shape,

    )

    head_config = dict(
        representation_size = 1024,
        num_classes = 14,
        stacked_fc = 2
    )
    model_train,model_rpn = build_model(backbone_config, neck_config,
                                                rpn_config,
                                                roi_config, head_config,anchors = anchors,
                                                image_shapes=image_shape)
    return model_train,model_rpn, anchors






                        