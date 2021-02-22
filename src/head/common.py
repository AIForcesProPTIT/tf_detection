import tensorflow as tf
from tensorflow import keras


def share_conv_module(features,norm=None, **kwargs ):
    
    if norm is None or norm == False:
        conv_share =  keras.layers.Conv2D(**kwargs)
        return [conv_share( feature) for feature in features]
    else:
        kwargs['use_bias']=False
        activation = kwargs.pop("activation",None)
        conv_share = keras.layers.Conv2D(**kwargs)
        features = [conv_share( feature) for feature in features]
        kwargs.pop("name")
        if isinstance(norm,bool):
            bn_share = keras.layers.BatchNormalization()
        elif isinstance(norm, keras.layers.Layer):
            bn_share = norm
        features = [bn_share( feature) for feature in features]

        if activation : 
            activation_share = keras.layers.Activation(activation=activation)
            return [activation_share( feature) for feature in features]
        return features

def reshape_to_valid_classifier( features, num_classes):
    shape = features.shape.as_list()# [None, h,w,c]
    assert shape[1] * shape[2] * shape[3] % num_classes == 0
    return keras.layers.Reshape(target_shape=(-1, num_classes)) (features)

def reshape_to_valid_reg( features):
    shape = features.shape.as_list()# [None, h,w,c]
    assert shape[1] * shape[2] * shape[3] % 4 == 0
    return keras.layers.Reshape(target_shape=(-1, 4))(features)



class FlattenFlexible(keras.layers.Reshape):
    def __init__(self, start_dim = 1, end_dim = -1,**kwargs):
        super().__init__(**kwargs)
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    def call(self, inputs):
        shape = inputs.shape.as_list()
        end_dim = self.end_dim
        if end_dim ==-1:end_dim = len(shape)-1
        new_shape = [0] * (len(shape) - (end_dim - self.start_dim))
        new_shape[0] =shape[0]
        for i in range(self.start_dim):new_shape[i] = shape[i] # start_dim(0,1)
        for i in range(end_dim + 1, len(shape)):new_shape[i] = shape[i] # 0
        f = 1
        for i in range(self.start_dim,end_dim+1):f = f * shape[i]
        new_shape[self.start_dim] =f

        return keras.layers.Reshape(target_shape=new_shape[1:])(inputs)
    def build(self, input_shape):
        shape = input_shape.as_list()
        end_dim = self.end_dim
        if end_dim ==-1:end_dim = len(shape)-1
        new_shape = [0] * len(shape) - (end_dim - self.start_dim)
        new_shape[0] =shape[0]
        for i in range(self.start_dim):new_shape[i] = shape[i] # start_dim(0,1)
        for i in range(end_dim + 1, len(shape)):new_shape[i] = shape[i] # 0
        f = 1
        for i in range(self.start_dim,end_dim+1):f = f * shape[i]
        new_shape[self.start_dim] = f
        seflf.target_shape = new_shape[1:]
        self.built = True
        
    def compute_output_shape(self, input_shape):
        shape = input_shape.as_list()
        end_dim = self.end_dim
        if end_dim ==-1:end_dim = len(shape)-1
        new_shape = [0] * len(shape) - (end_dim - self.start_dim)
        new_shape[0] =shape[0]
        for i in range(self.start_dim):new_shape[i] = shape[i] # start_dim(0,1)
        for i in range(end_dim + 1, len(shape)):new_shape[i] = shape[i] # 0
        f = 1
        for i in range(self.start_dim,end_dim+1):f = f * shape[i]
        new_shape[self.start_dim] = f
        return new_shape
    def get_config(self):
        config = super().get_config()
        config.update({
            'start_dim':self.start_dim,
            'end_dim':self.end_dim
        })

        # batch,100,7,7,100
        # endim = 5 start_dim = 2 
        # (batch,100,-1) = 3 = 5-(2) = 3