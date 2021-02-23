import tensorflow as tf
from tensorflow import keras



from src.backbone.resnet_features import get_resnet
from src.neck.build_neck import build_from_config
from src.head.anchor_based_head import build_anchor_based_head
from src.one_stage import build_head
from src.common.decoder import DecodePredictions
from src.head.roi_head import MultiScaleRoIAlign
from src.head.common import FlattenFlexible
from src.common.assign_target import AssignTargetTF

class ModelTrainWraperTwoStage(keras.Model):
    def __init__(self, *args,assign_target = None, rpn_train_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        if assign_target is None:
            assign_target = AssignTargetTF()
        self.assign_target = assign_target
        self.rpn_train_only = rpn_train_only

    def train_step(self, data):
        images, matched_gt_boxes, matched_gt_labels,\
        mask_bboxes,mask_labels,bboxes, labels = data
        # first train rpn
        y_train_rpn = (matched_gt_boxes, matched_gt_labels,\
                    mask_bboxes,mask_labels)
        
        
        with tf.GradientTape() as tape:
            
            if self.rpn_train_only is False:
                convs, regs, decoder_rpn,\
                    rpn_convs, rpn_regs = self(images, training=True)
                loss_rpn = self.loss.get("rpn_loss_func")(
                  y_train_rpn, (rpn_convs, rpn_regs))
                (target_regressiones,\
                target_clasifications,\
                mask_regressiones,\
                mask_clasifications) = self.assign_target(bboxes, labels, decoder_rpn)

                loss_stage_two = self.loss.get("head_loss_func")(
                    (target_regressiones, target_clasifications, mask_regressiones, mask_clasifications),
                    (convs, regs)
                )
                loss_stable = loss_rpn + loss_stage_two
                loss_stable = [i * tf.cast( tf.logical_not(tf.logical_or(tf.math.is_nan(i), tf.math.is_inf(i))), tf.float32) for i in loss_stable  ]
                total_loss = tf.math.reduce_sum(loss_stable)
                total_loss = total_loss 
                trainable_vars = self.trainable_variables
            else:
                rpn_convs, rpn_regs = self(images, training=True)
                loss_rpn = self.loss.get("rpn_loss_func")(
                    y_train_rpn, (rpn_convs, rpn_regs))

                loss_stable = loss_rpn
                loss_stable = [i * tf.cast( tf.logical_not(tf.logical_or(tf.math.is_nan(i), tf.math.is_inf(i))), tf.float32) for i in loss_stable  ]
                total_loss = tf.math.reduce_sum(loss_stable)
                total_loss = total_loss 
                trainable_vars = self.trainable_variables
                loss_stage_two = [0.,0.]
            
            # modifier loss here : scale loss ... 
            scaled_loss = total_loss

        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = scaled_gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss_cls':loss_rpn[1], 'loss_reg':loss_rpn[0],'loss_head_reg':loss_stage_two[0],
                'loss_head_cls': loss_stage_two[1],'total_loss':total_loss}

def build_model(backbone_config:dict,
                    neck_config: dict,
                    rpn_config : dict  ,
                    roi_config : dict,
                    head_config: dict ,
                    anchors = None,
                    image_shapes = (512, 512, 3)):

    
    # backbone
    backbone = build_backbone(
        backbone_config.pop("name","none",), image_inputs=image_shapes
    )

    # neck : fpn

    necks, config_neck = build_neck(backbone.outputs, neck_config.get("build_node"), True)

    # rpn return (convs, regs)
    rpn = build_head(
        necks,
        **rpn_config,
    )
    # print(rpn)
    rpn_config_decoder = dict(
        anchors=anchors,
        image_shape=image_shapes,
        name="decoder_rpn_layer",
    )
    rpn_config_decoder.update(**rpn_config)
    decoder_rpn_layer = DecodePredictions(**rpn_config_decoder)
    
    num_boxes = decoder_rpn_layer.max_detections_training

    decoder_rpn = decoder_rpn_layer([rpn[0], rpn[1]])
    
    head_crop_rpn = MultiScaleRoIAlign(name="multiScaleRoi", **roi_config)((decoder_rpn[0],necks))


    head_crop_rpn_shape = head_crop_rpn.shape.as_list()[2:] 
    
    f = 1
    for i in head_crop_rpn_shape:f = f * i

    head_flatten = tf.keras.layers.Reshape(target_shape=(num_boxes,f  ), name = "flatten")(head_crop_rpn)

    
    convs,regs = build_flatten_crop(head_flatten, **head_config)

   

    model_train = ModelTrainWraperTwoStage(inputs = backbone.inputs, outputs = [convs, regs, decoder_rpn[0], rpn[0],rpn[1]])
    model_rpn = ModelTrainWraperTwoStage(inputs = backbone.inputs, outputs = [rpn[0],rpn[1]])
    model_rpn.rpn_train_only = True
    return model_train,model_rpn
    


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





def build_flatten_crop(head_flatten, representation_size = 1024, num_classes = 14, stacked_fc = 2,**kwargs):
    activation = kwargs.get("activation","relu")
    norm = kwargs.get("norm",False)
    if norm is False:
        for i in range(stacked_fc):
            head_flatten = keras.layers.Dense(representation_size, activation=activation)(head_flatten)
    else:
        for i in range(stacked_fc):
            head_flatten = keras.layers.Dense(representation_size, activation=None, use_bias=False)(head_flatten)
            head_flatten = keras.layers.BatchNormalization()(head_flatten)
            head_flatten = keras.layers.Activation(activation)(head_flatten)

    convs = keras.layers.Dense(num_classes, activation='sigmoid')(head_flatten)
    regs = keras.layers.Dense(num_classes * 4, activation=None)(head_flatten)

    return convs,regs

import torchvision.models.detection.faster_rcnn

