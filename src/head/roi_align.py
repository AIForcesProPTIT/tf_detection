# roi_head : get boxes from anchor_base_head (rpn layers) -> crop_boxes -> cls_pred,box_pred

from tensorflow import keras
import tensorflow as tf
import numpy as np


def log2_graph(x):return tf.math.log(x) / tf.math.log(2.)

class RoiAlign(keras.layers.Layer):
    """Roifeatures to roi
        inputs : features_map : [List(Batch_size, H',W', C)] features_maps[i] = level[i]
                 bboxes : (batch_size, num_boxes,(x1, y1, x2, y2)) in normailized
        outputs : Tensor (batch_size,num_boxes,*crop_size,,depth]
    """
    def __init__(self,crop_size=(7, 7), canonical_size = 56., min_level=0, max_level=4, method ='bilinear',**kwargs): # 0->4 :C2, C3.C4.C5,max_poll
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'crop_size':self.crop_size,
            'min_level':self.min_level,
            'max_level':self.max_level,
            'canonical_size':self.canonical_size
        })
    
    def call(self, boxes, feature_maps, image_shape):
        
        x1, y1, x2, y2 = tf.split(boxes, 4, axis=2) # batch,num_boxes
        bboxes = tf.concat([y1, x1, y2, x2], axis = -1) # for tf format :v
        h = y2 - y1 
        w = x2 - x1
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



def test_roi_align():
    
    xy = np.random.uniform(size=(2, 15, 2)) * 256. # batch_size ,num_bboxes, 4
    x2y2 = np.random.uniform(size=(2, 15, 2)) * 256 + xy
    bboxes = np.concatenate((xy,x2y2), axis=-1) /512.
    bboxes[0,0,:] = np.array([0.,0.,1.,1.])
    feature_maps = [tf.random.normal(shape=(2, 512 // 2**i, 512 // 2**i , 3)) for i in range(2,7)]

    bboxes = tf.convert_to_tensor(bboxes, tf.float32)
    image_shape = [512., 512.]
    roi_align = RoiAlign(crop_size=(8,8))
    out = roi_align(bboxes, feature_maps, image_shape)
    assert all( map(lambda x:x[0] ==x[1], list(zip(out.shape.as_list() , [2, 15 , 8, 8, 3]) ) ))
    return out,bboxes



# Roi_aligin -> []