import tensorflow as tf
from tensorflow import keras

from src.backbone.resnet_features import get_resnet
from src.neck.build_neck import build_from_config
from src.head.anchor_based_head import build_anchor_based_head
from src.common.decoder import DecodePredictions
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

def build_head(necks, name_head = 'retina', **config):
    """features,
        num_classes=1,
        num_anchors = 9,
        feat_channels=256,
        stacked_convs=4,
        **kwargs
    """
    if name_head == 'anchor_based_head':
        return build_anchor_based_head(necks, **config)

def build_model(
    backbone_config:dict,
    neck_config :dict,
    head_config: dict,
    anchor,
    inference_config = dict(
        confidence_threshold=0.15,
        nms_iou_threshold=0.5,
        max_detections_per_class=200,
        max_detections=200,
        max_detections_per_class_training=200,
        max_detections_training=200,
        box_variance=[0.1, 0.1, 0.2, 0.2],
    ),
    **kwargs):
    
    backbone_name = backbone_config.get("backbone_name", None)
    image_inputs  = backbone_config.get("inputs_shape", ( 512, 512, 3))

    backbone = build_backbone(backbone_name=backbone_name, image_inputs = image_inputs)

    necks, config_neck = build_neck(backbone.outputs, neck_config.get("build_node"), True)

    head = build_head(  necks,
                        name_head = head_config.pop("head_name","anchor_based_head"),
                        **head_config )

    
    model_train =  ModelTrainWraper(anchor_generator=anchor,inputs = backbone.inputs, outputs = head)

    # decoder = DecodePredictions(
    #     **inference_config,
    #     anchors=anchor,
    # )([head[0], head[1]])


    

    return model_train
    

class ModelTrainWraper(keras.Model):
    
    def __init__(self, *args, anchor_generator = None,**kwargs):
        super().__init__(*args,**kwargs)
        self.anchors = anchor_generator
        self.ap = 50.
    def train_step(self, data):
        images, matched_gt_boxes, matched_gt_labels,\
        mask_bboxes,mask_labels= data
        y_true = (matched_gt_boxes, matched_gt_labels,\
                    mask_bboxes,mask_labels)
        with tf.GradientTape() as tape:
            
            convs, regs = self(images, training=True)
            loss = self.loss(
                  y_true, (convs, regs))
#             print(loss)
            total_loss = loss[0] * self.ap + loss[1] 
            
            trainable_vars = self.trainable_variables
            
            scaled_loss = total_loss
            
#         learning_rate = float(tf.keras.backend.get_value(self.optimizer.lr))
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = scaled_gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.compiled_metrics.update_state(y_true, (convs,regs))
        return {'loss_cls':loss[1], 'loss_reg':loss[0], 'total_loss':total_loss}
    






