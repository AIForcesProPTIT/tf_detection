from tensorflow import keras
import tensorflow as tf


from src.one_stage import build_model

# config example for retina_net

backbone_config = {
    'name':'resnet_v1_50',
    'inputs_shape':(1024, 1024, 3)
}

neck_config = {
    'name':'fpn_network',
    'build_node': 
        [
            ('c5_branch_2' , [ {'4' : 'through' , 'kwargs':{} },{'type':'identity'}]),
            ('c4_branch_2' , [ {'3' : 'through'}, {'5' : 'up_sample'},{'type':'add'}]),
            ('c3_branch_2' , [ {'2' : 'through'}, {'6': 'up_sample'}, {'type':'add'}]),
            ('c2_branch_2' , [ {'1' : 'through'}, {'7': 'up_sample'},{'type':'add'}]),
            ('c6_branch_1' , [ {'4' : 'down_sample'},{'type':'identity'}] ),
            ('c7_branch_1' , [ {'9' : 'down_sample'},{'type':'identity'}] ),
            ('return_node' , [7, 6, 5, 9, 10] )
        ]
    # Node will build likes : 
    #"""
        #c6  9  ->  c7   10
        # |
        #c5  4   -> c5'   5
        #            |
        #c4  3   ->  c4'  6
        #            |
        #c3  2   ->  c3'  7
        #            |
        #c2  1   ->  c2'  8

        #c1  0 
        #return [c3:7,c4:6,c5:5,c6:9,c7:10]  = [7, 6, 5, 9, 10]
    #"""
    # some config inside kwargs : build convsModule : example kwargs :{"filters":512, "activation":'relu',"norm":True || your custom norm..} ...
    # todos : make convsModule more stable with SeperateConvs, custom Norm, custom activations ...
    # todos : add more examples for build_node with BiFPN, PA-FPN,... 
}

head_config = {
    'head_name':'retina',
    "num_classes":14,
    "num_anchors":9,
    "feat_channels":256,
    "stacked_convs":4
}
# just provide typical retina_head:
# todos add rpn_head ( faster-rcnn )

model = build_model(
    backbone_config,
    neck_config,
    head_config
)
