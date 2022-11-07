# from keras.models import *
# from keras.layers import *
# from types import MethodType
# # from project.resnet50 import get_resnet50_encoder, IMAGE_ORDERING
# from models.functions import predict
#
#
# def get_segmentation_model(input, output):
#
#     img_input = input
#     o = output
#
#     o_shape = Model(img_input, o).output_shape
#     i_shape = Model(img_input, o).input_shape
#
#     output_height = o_shape[1]
#     output_width = o_shape[2]
#     input_height = i_shape[1]
#     input_width = i_shape[2]
#     n_classes = o_shape[3]
#     o = (Reshape((output_height*output_width, -1)))(o)
#
#     o = (Activation('softmax'))(o)
#     model = Model(img_input, o)
#     model.output_width = output_width
#     model.output_height = output_height
#     model.n_classes = n_classes
#     model.input_height = input_height
#     model.input_width = input_width
#     model.model_name = ""
#
#     model.predict_segmentation = MethodType(predict, model)
#
#     return model
#
#
# def crop(o1, o2, i):
#     o_shape2 = Model(i, o2).output_shape
#
#     output_height2 = o_shape2[1]
#     output_width2 = o_shape2[2]
#
#     o_shape1 = Model(i, o1).output_shape
#
#     output_height1 = o_shape1[1]
#     output_width1 = o_shape1[2]
#
#     cx = abs(output_width1 - output_width2)
#     cy = abs(output_height2 - output_height1)
#
#     if output_width1 > output_width2:
#         o1 = Cropping2D(cropping=((0, 0), (0, cx)),
#                         data_format=IMAGE_ORDERING)(o1)
#     else:
#         o2 = Cropping2D(cropping=((0, 0), (0, cx)),
#                         data_format=IMAGE_ORDERING)(o2)
#
#     if output_height1 > output_height2:
#         o1 = Cropping2D(cropping=((0, cy), (0, 0)),
#                         data_format=IMAGE_ORDERING)(o1)
#     else:
#         o2 = Cropping2D(cropping=((0, cy), (0, 0)),
#                         data_format=IMAGE_ORDERING)(o2)
#
#     return o1, o2
#
#
# def fcn_8(n_classes, encoder=get_resnet50_encoder, input_height=416,
#           input_width=608, channels=3):
#     img_input, levels = encoder(
#         input_height=input_height, input_width=input_width, channels=channels)
#     [f1, f2, f3, f4, f5] = levels
#
#     o = f5
#
#     o = (Conv2D(4096, (7, 7), activation='relu',
#                 padding='same', data_format=IMAGE_ORDERING))(o)
#     o = Dropout(0.5)(o)
#     o = (Conv2D(4096, (1, 1), activation='relu',
#                 padding='same', data_format=IMAGE_ORDERING))(o)
#     o = Dropout(0.5)(o)
#
#     o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
#                 data_format=IMAGE_ORDERING))(o)
#     o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(
#         2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
#
#     o2 = f4
#     o2 = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
#                  data_format=IMAGE_ORDERING))(o2)
#
#     o, o2 = crop(o, o2, img_input)
#
#     o = Add()([o, o2])
#
#     o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(
#         2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
#     o2 = f3
#     o2 = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
#                  data_format=IMAGE_ORDERING))(o2)
#     o2, o = crop(o2, o, img_input)
#     o = Add(name="seg_feats")([o2, o])
#
#     o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(
#         8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
#
#     model = get_segmentation_model(img_input, o)
#     model.model_name = "fcn_8"
#     return model
