


import tensorflow as tf
from tensorflow import keras
import functools

from collections import OrderedDict

def gen_fpn_config_use_p6_p7():
    ''' 
        c6   9  ->  c7   10
         |
        c5  4   -> c5'   5
                    |
        c4  3   ->  c4'  6
                    |
        c3  2   ->  c3'  7
                    |
        c2  1   ->  c2'  8

        c1  0 
    '''
    config= [
        
            ('c5_branch_2' , [ {'4' : 'through' , 'kwargs':{} },{'type':'identity'}]),
            ('c4_branch_2' , [ {'3' : 'through'}, {'5' : 'up_sample'},{'type':'add'}]),
            ('c3_branch_2' , [ {'2' : 'through'}, {'6': 'up_sample'}, {'type':'add'}]),
            ('c2_branch_2' , [ {'1' : 'through'}, {'7': 'up_sample'},{'type':'add'}]),
            ('c6_branch_1' , [ {'4' : 'down_sample'},{'type':'identity'}] ),
            ('c7_branch_1' , [ {'9' : 'down_sample'},{'type':'identity'}] ),
            ('return_node' , [7, 6, 5, 9, 10] )
    ]
    
    return config


