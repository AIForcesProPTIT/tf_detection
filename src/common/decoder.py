import tensorflow as tf
from src.common.box_utils import coordinates_to_center, center_to_coordinates
class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=200,
        max_detections=200,
        max_detections_per_class_training=200,
        max_detections_training=200,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        anchors=None,
        name = "DecodePredictions",
        **kwargs
    ):
        super(DecodePredictions, self).__init__(name=name)
      
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self.max_detections_per_class_training = max_detections_per_class_training
        self.max_detections_training=max_detections_training
        self._anchor_box = anchors
        assert self._anchor_box is not None
       
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        # anchor_boxes (None, numanchors,4)
        # box_predictions = (batchsize, numanchors,4)
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
       

        boxes_transformed = center_to_coordinates(boxes) # batch_size, num_anchors,4
        
        return boxes_transformed


    def call(self,inputs, training=None):
        head_classifier, head_regression = inputs
        anchor_boxes =tf.convert_to_tensor(self._anchor_box,tf.float32)

        box_predictions = head_regression # shape = (batch, numboxes, 4)
        cls_predictions = tf.nn.sigmoid(head_classifier) # (batch, numboxes, num_classes) # 

        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        
        max_detections_per_class=self.max_detections_per_class
        max_detections = self.max_detections
        if training is True:
            max_detections = self.max_detections_training
            max_detections_per_class = self.max_detections_per_class_training
        out= tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            max_detections_per_class,
            max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
        return out.nmsed_boxes,out.nmsed_scores,out.nmsed_classes,out.valid_detections
    def get_config(self):
        config = super().get_config()
        config.update({
       
            "confidence_threshold":self.confidence_threshold,
            "nms_iou_threshold":self.nms_iou_threshold,
            "max_detections_per_class":self.max_detections_per_class,
            "max_detections":self.max_detections,
            "_box_variance":self._box_variance,
            "_anchor_box":self._anchor_box,
            "max_detections_training":self.max_detections_training,
            "max_detections_per_class_training":self.max_detections_per_class_training
        })
        return config

    def compute_output_shape(self, inputs_shape):
        return (
            (None,4),
            (None,),
            (None,),
            (1,)
        )