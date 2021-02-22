from src.head.roi_align import RoiAlign,log2_graph
from src.common.assign_target import AssignTargetTF

from tensorflow import keras

import tensorflow as tf

from src.common.box_utils import coordinates_to_center, center_to_coordinates
# rpn -> roihead -> compute loss 
# trong roihead

class MultiScaleRoIAlign(keras.layers.Layer):
    """Roifeatures to roi
        inputs : features_map : [List(Batch_size, H',W', C)] features_maps[i] = level[i]
                 bboxes : (batch_size, num_boxes,(x1, y1, x2, y2)) in normailized
        outputs : Tensor (batch_size,num_boxes,*crop_size,,depth]
    """
    def __init__(self,crop_size=(7, 7), canonical_size = 56.,
                    min_level=0, max_level=4,
                    method ='bilinear',
                    image_shape=(1024,1024,3),
                    **kwargs): # 0->4 :C2, C3.C4.C5,max_poll
        """canonical_size = 56: size <=56 : map to c2 (usually c2 map to size_anchor 32 so canonical_size = 56)
        """
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.min_level = min_level
        self.max_level = max_level
        self.canonical_size = canonical_size
        self.method = method
        assert min_level < max_level
        assert all(map(lambda x:x>0, crop_size))
        
        self.image_shape = image_shape


    def get_config(self):
        config = super().get_config()
        config.update({
            'crop_size':self.crop_size,
            'min_level':self.min_level,
            'max_level':self.max_level,
            'canonical_size':self.canonical_size,
            "image_shape":self.image_shape,
            "method":self.method
            
        })

    def call(self, inputs):
        bboxes, feature_maps=inputs
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=-1) # batch,num_boxes
        
        # clip boxes 

        width = self.image_shape[1]
        height = self.image_shape[0]

        x1 = tf.clip_by_value(x1, 0, width  )
        y1 = tf.clip_by_value(y1, 0, height )
        x2 = tf.clip_by_value(x2, 0, width  )
        y2 = tf.clip_by_value(y2, 0, height )
        x1 = x1 / width
        x2 = x2 / width
        y1 = y1 / height
        y2 = y2 / height

        boxes = tf.concat([y1, x1, y2, x2], axis = -1) # for tf format :v
        h = y2 - y1 
        w = x2 - x1
        image_shape = self.image_shape
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32) # scalaer 


        roi_level = log2_graph(tf.sqrt(h * w) / (self.canonical_size / tf.sqrt(image_area))) # batch_,num_boxes


        roi_level = tf.minimum(self.max_level, tf.maximum(
            self.min_level, tf.cast(tf.round(roi_level), tf.int32)))

        roi_level = tf.squeeze(roi_level, 2)
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.crop_size,
                method=self.method))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.crop_size + (input_shape[1][-1], ) # batch, num_boxes + (crop_size) + depth

