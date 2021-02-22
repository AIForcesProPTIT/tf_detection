# Object Detection With Tensorflow

1. backbone: Resnet_V*
2. neck : FPN
3. head : anchor_base retina_head typical (shoulde change name).**done**

## Todos : 

0. add two stage detector : faster-rcnn **done**
1. add more backbone : some backbone already in keras, and some not : CSP-backbone(Yolo), efficientNet,..
2. Create ConvsModule stable for config neck.
3. add config more exsample for BiFPN,PAN, NAS-FPN ,ASFF,SFAM... (just add config example, code generator can build all). # example FPN in src/neck/config.py **done**
4. add anchor_free.
5. add coco metrics.
6. create tf augmentation to touch the best performance tf.data. (now use albumentations with from generators:api).
