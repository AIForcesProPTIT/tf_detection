import tensorflow as tf

from tensorflow import keras

from src.neck.config import gen_fpn_config_use_p6_p7
from src.neck.common import build_down_sample,build_through_conv,build_up_sample

import copy
def build_from_config(inputs, config = None):

    if config is None:
        config = gen_fpn_config_use_p6_p7()
    else :
        config = copy.deepcopy(config)

    # assert len(inputs) >= 5, 'len inputs shoulde greater or equal 5 [c1,c2,c3,c4,c5]'

    feats = inputs
    config_ = copy.deepcopy(config)

    for index_node, (node_name,node) in enumerate(config):
        if node_name == 'return_node':
            return [feats[i] for i in node],config_
        node_inputs = []

        for index_shortcut,shortcut in enumerate(node[:-1]): # last node is meger type
            for c,index in enumerate(shortcut.keys()):
                if index == 'kwargs':continue # information of short_cut
                
                if shortcut[index] == 'through':
                    kwargs_config = shortcut.get("kwargs", {})
                    kwargs  = dict(
                        filters = 256,
                        kernel_size = (3,3),
                        strides = (1,1),
                        padding = 'SAME',
                        activation='relu',
                        name = node_name + '_through_' + str(c)
                    )
                    #overide
                    for key in kwargs_config.keys():
                        kwargs[key] = kwargs_config[key]

                    config_[index_node][1][index_shortcut]['kwargs'] = kwargs

                    node_inputs.append(build_through_conv(feats[int(index)], **kwargs))
                elif shortcut[index] == 'up_sample':
                    kwargs_config = shortcut.get("kwargs", {})
                    
                    kwargs  = dict(
                        filters = 256,
                        kernel_size = (3,3),
                        strides = (1,1),
                        padding = 'SAME',
                        activation='relu',
                        name = node_name + '_up_sample_' + str(c)
                    )
                    #overide
                    for key in kwargs_config.keys():
                        kwargs[key] = kwargs_config[key]
                    config_[index_node][1][index_shortcut]['kwargs'] = kwargs
                    node_inputs.append( build_up_sample (feats[int(index)], **kwargs))
                elif shortcut[index] == 'down_sample':
                    kwargs_config = shortcut.get("kwargs", {})
                    
                    kwargs  = dict(
                        filters = 256,
                        kernel_size = (3,3),
                        strides = (1,1),
                        padding = 'SAME',
                        activation='relu',
                        name = node_name + '_down_sample_' + str(c)
                    )
                    #overide
                    for key in kwargs_config.keys():
                        kwargs[key] = kwargs_config[key]
                    config_[index_node][1][index_shortcut]['kwargs'] = kwargs
                    node_inputs.append( build_down_sample( feats[int(index)], **kwargs))
                else:
                    raise Exception(f"not support type shortcut {shortcut[index]}")
        
       
        
        type_meger =  node[-1].get("type",None)

        if type_meger == None:
            type_meger = 'add'
            config_[index_node][1][-1]['type'] = 'add'
        if type_meger == 'add':
            config_[index_node][1][-1]['type'] = 'add'
            try:
                feats.append ( keras.layers.Add(name = node_name + '_out') ( node_inputs) )
            except:
                print(node_inputs)
                print(shortcut)
        elif type_meger == 'identity':
            feats = feats + node_inputs
            config_[index_node][1][-1]['type'] = 'identity'
    return feats,config_

    