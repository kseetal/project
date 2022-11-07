import tensorflow as tf
from math import ceil
from resnet50 import ResNet
from keras import layers
from keras.layers import *
from keras.engine.training import Model

def pspnet_101_voc12():

    model_config = {
        "input_height": 473,
        "input_width": 473,
        "n_classes": 21,
        "model_class": "pspnet_101",
    }

    model = pspnet_101(model_config["n_classes"], model_config["input_height"],
            model_config["input_width"], 3)


    return model


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        try:
            resized = tf.image.resize(inputs, [new_height, new_width])
        except AttributeError:
            resized = tf.image.resize_images(inputs, [new_height, new_width],
                                             align_corners=True)
        return resized


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (473, 473):
        kernel_strides_map = {1: 60,
                              2: 30,
                              3: 20,
                              6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90,
                              2: 45,
                              3: 30,
                              6: 15}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def _build_pspnet(nb_classes, resnet_layers, input_shape,
                  activation='softmax', channels=3):

    inp = Input((input_shape[0], input_shape[1], channels))

    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    o_shape = x.shape
    i_shape = inp.shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    x = (Reshape((output_height * output_width, -1)))(x)

    x = (Activation('softmax'))(x)
    model = Model(inp, x)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.seg_feats_layer_name = "conv5_4"

    return model


def pspnet_101(n_classes,  input_height=473, input_width=473, channels=3):

    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=n_classes,
                          resnet_layers=101,
                          input_shape=input_shape, channels=channels)
    model.model_name = "pspnet_101"
    return model