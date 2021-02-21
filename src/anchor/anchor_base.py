import tensorflow as tf
from tensorflow import keras
import numpy as np
import six
from abc import ABC,abstractclassmethod,ABCMeta,abstractmethod


class AnchorGenerator(keras.layers.Layer):
    
    def __init__(self,
                        sizes=((128, 256, 512),),
                        aspect_ratios=((0.5, 1.0, 2.0),),
                        **kwargs):
        super().__init__(**kwargs)

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None

    
    def get_config(self):
        config = super(AnchorGenerator,self).get_config()
        config.update({
            'sizes':self.sizes,
            'aspect_ratios':self.aspect_ratios,
            'cell_anchors':self.cell_anchors
        })
        return config
    

    def generate_anchors(self, scales, aspect_ratios, dtype=tf.float32):
        # type: (List[int], List[float], int, Device) -> Tensor  # noqa: F821
        scales = tf.convert_to_tensor(scales, dtype=dtype)
        aspect_ratios = tf.convert_to_tensor(aspect_ratios, dtype=dtype)
        h_ratios = tf.math.sqrt(aspect_ratios)

        w_ratios =tf.cast( tf.math.divide(tf.identity(1.),h_ratios), dtype=dtype)


        ws = tf.reshape( tf.expand_dims( w_ratios, 1) * tf.expand_dims(scales, 0), (-1, ) )

        hs = tf.reshape( tf.expand_dims(h_ratios, 1 ) * tf.expand_dims(scales, 0), (-1,))

        base_anchors = tf.math.divide(tf.stack([-ws, -hs, ws, hs], axis=1),tf.identity(2.))
        return tf.math.round(base_anchors)

    def set_cell_anchors(self, dtype=tf.float32):
        # type: (int, Device) -> None  # noqa: F821
        if self.cell_anchors is not None:
            if self.cell_anchors.dtype == dtype:
                return
            else:
                self.cell_anchors = tf.cast(self.cell_anchors,dtype=dtype)
            return 
        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                dtype,
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
#         print(cell_anchors)
        self.cell_anchors =tf.RaggedTensor.from_tensor(cell_anchors)
    
    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        assert len(grid_sizes) == len(strides) == cell_anchors.shape.as_list()[0]

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = tf.range(
                0, grid_width, dtype=tf.float32
            ) * stride_width
            shifts_y = tf.range(
                0, grid_height, dtype=tf.float32
            ) * stride_height

            shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x)
            shift_x =tf.reshape( shift_x, (-1, ) )
            shift_y =tf.reshape( shift_y, (-1,) )

            shifts = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
               tf.reshape((tf.reshape( shifts, (-1, 1, 4)) + tf.reshape( base_anchors, (1, -1, 4) )), (-1, 4) )
            )

        return anchors

    def call(self, image_list, feature_map_shapes):
        # type: (ImageList, List[Tensor]) -> List[Tensor]

        grid_sizes = list([feature_map[-3:-1] for feature_map in feature_map_shapes])
        image_size = image_list.get_shape()[-3:-1]

        dtype = image_list.dtype
        
        strides = [[(image_size[0] // g[0]),
                    (image_size[1] // g[1])] for g in grid_sizes]

        self.set_cell_anchors(dtype)

        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)

        anchors = tf.concat(anchors_over_all_feature_maps,axis=0)
        return anchors

# [(H),C,W]

def test_anchor():
    import os
    # previous_os = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    aspect_ratios = ((0.5,1.0),(0.5,1.0))
    anchors_sizes = ((10,),(20,))
    anchor_generator = AnchorGenerator(anchors_sizes,aspect_ratios)
    image = tf.random.normal(shape=(1,40,40,3),dtype=tf.float32)
    image.shape[-3:-1]
    feature_maps = [tf.random.normal(shape=(1,10,10,12)),tf.random.normal(shape=(1,20,20,6))][::-1]
    x = anchor_generator(image,feature_maps)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    return x

# print(test_anchor())