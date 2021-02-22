import tensorflow as tf
from tensorflow import keras

from src.backbone.resnet_features import get_resnet
from src.neck.build_neck import build_from_config
from src.head.anchor_based_head import build_anchor_based_head
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

def build_head(necks, name_head = 'retina', config  = {}):
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
    head_config: dict,**kwargs):
    
    backbone_name = backbone_config.get("backbone_name", None)
    image_inputs  = backbone_config.get("inputs_shape", ( 512, 512, 3))

    backbone = build_backbone(backbone_name=backbone_name, image_inputs = image_inputs)

    necks, config_neck = build_neck(backbone.outputs, neck_config.get("build_node"), True)

    head = build_head(  necks,
                        name_head = head_config.pop("head_name","retina"),
                        config = head_config )

    
    return backbone.inputs, head

    

tf.image.generate_bounding_box_proposals
    






