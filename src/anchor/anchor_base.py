import tensorflow as tf
from tensorflow import keras
import numpy as np
import six
from abc import ABC,abstractclassmethod,ABCMeta,abstractmethod
from typing import Text, Tuple, Union

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
            ) * tf.cast(stride_width,tf.float32)
            shifts_y = tf.range(
                0, grid_height, dtype=tf.float32
            ) * tf.cast(stride_height,tf.float32)

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

    def call(self, image_size, feature_map_shapes):
        # type: (ImageList, List[Tensor]) -> List[Tensor]

        grid_sizes = list([feature_map[-3:-1] for feature_map in feature_map_shapes])
        
        dtype = tf.float32
        
        strides = [[(image_size[0] // g[0]),
                    (image_size[1] // g[1])] for g in grid_sizes]
        print(strides,image_size,feature_map_shapes,grid_sizes)
        self.set_cell_anchors(dtype)

        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)

        anchors = tf.concat(anchors_over_all_feature_maps,axis=0)
        return anchors

def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]],
                   max_level: int):
  """Get feat widths and heights for all levels.
  Args:
    image_size: A integer, a tuple (H, W), or a string with HxW format.
    max_level: maximum feature level.
  Returns:
    feat_sizes: a list of tuples (height, width) for each level.
  """
  
  feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
  feat_size = image_size
  for _ in range(1, max_level + 1):
    feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
    feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
  return feat_sizes

class Anchors():
  """Multi-scale anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_scale, image_size):
    """Constructs multiscale anchors.
    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of representing the aspect ratio anchors added
        on each level. For instances, aspect_ratios = [1.0, 2.0, 0..5]
        adds three anchors on each level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level. Or a list, one value per layer.
      image_size: integer number or tuple of integer number of input image size.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    if isinstance(anchor_scale, (list, tuple)):
      assert len(anchor_scale) == max_level - min_level + 1
      self.anchor_scales = anchor_scale
    else:
      self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)
    self.image_size = image_size
    self.feat_sizes = get_feat_sizes(image_size,max_level)
    self.config = self._generate_configs()
    self.boxes = self._generate_boxes()

  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    anchor_configs = {}
    feat_sizes = self.feat_sizes
    for level in range(self.min_level, self.max_level + 1):
      anchor_configs[level] = []
      for scale_octave in range(self.num_scales):
        for aspect in self.aspect_ratios:
          anchor_configs[level].append(
              ((feat_sizes[0]['height'] / float(feat_sizes[level]['height']),
                feat_sizes[0]['width'] / float(feat_sizes[level]['width'])),
               scale_octave / float(self.num_scales), aspect,
               self.anchor_scales[level - self.min_level]))
    return anchor_configs

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes_all = []
    for _, configs in self.config.items():
      boxes_level = []
      for config in configs:
        stride, octave_scale, aspect, anchor_scale = config
        base_anchor_size_x = anchor_scale * stride[1] * 2**octave_scale
        base_anchor_size_y = anchor_scale * stride[0] * 2**octave_scale
        if isinstance(aspect, list):
          aspect_x, aspect_y = aspect
        else:
          aspect_x = np.sqrt(aspect)
          aspect_y = 1.0 / aspect_x
        anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0
        anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0

        x = np.arange(stride[1] / 2, self.image_size[1], stride[1])
        y = np.arange(stride[0] / 2, self.image_size[0], stride[0])
        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)

        boxes = np.vstack((xv - anchor_size_y_2, yv - anchor_size_x_2,
                           xv + anchor_size_y_2, yv + anchor_size_x_2))
        boxes = np.swapaxes(boxes, 0, 1)
        boxes_level.append(np.expand_dims(boxes, axis=1))
      # concat anchors on the same level to the reshape NxAx4
      boxes_level = np.concatenate(boxes_level, axis=1)
      boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all) # batch_4
    # anchor_boxes = 
    anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
    return anchor_boxes

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

# def test_anchor():
#     import os
#     # previous_os = os.environ["CUDA_VISIBLE_DEVICES"]
#     os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#     aspect_ratios = ((0.5,1.0),(0.5,1.0))
#     anchors_sizes = ((10,),(20,))
#     anchor_generator = AnchorGenerator1(anchors_sizes,aspect_ratios)
#     image = tf.random.normal(shape=(1,40,40,3),dtype=tf.float32)
#     # image.shape[-3:-1]
#     feature_maps = [tf.random.normal(shape=(1,10,10,12)),tf.random.normal(shape=(1,20,20,6))][::-1]
#     x = anchor_generator(image.shape.as_list()[1:],[i.shape.as_list() for  i in feature_maps])

#     anchor_2 = CustomAnchorGenerator(anchors_sizes, aspect_ratios)
#     feature_maps = [torch.from_numpy(feature_map.numpy()).permute(0,3,1,2) for feature_map in feature_maps]
#     y = anchor_2(torch.zeros(1,3,40,40),feature_maps)
#     # print(x,y,sep="-"*50 + "\n")
#     # print(y.shape)
#     print(np.mean( x.numpy()-y.numpy()))
#     os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#     return x

# print(test_anchor())