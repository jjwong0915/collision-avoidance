from __future__ import absolute_import, division, print_function
import os, time, cv2, pickle, struct, re, random
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
from PIL import Image
from random import shuffle
from enum import Enum

from keras.models import Sequential, Model, load_model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, UpSampling2D
from keras.layers import LeakyReLU, ZeroPadding2D, Add, merge, DepthwiseConv2D
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.applications.mobilenetv2 import MobileNetV2
from keras import regularizers, initializers


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _depthwise_conv_block_v2(inputs,
                             pointwise_conv_filters,
                             alpha,
                             expansion_factor,
                             depth_multiplier=1,
                             strides=(1, 1),
                             bn_epsilon=1e-3,
                             bn_momentum=0.999,
                             block_id=1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)
    depthwise_conv_filters = _make_divisible(input_shape[channel_axis] *
                                             expansion_factor)
    pointwise_conv_filters = _make_divisible(pointwise_conv_filters * alpha)

    if depthwise_conv_filters > input_shape[channel_axis]:
        x = Conv2D(depthwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   name='conv_expand_%d' % block_id)(inputs)
        x = BatchNormalization(axis=channel_axis,
                               momentum=bn_momentum,
                               epsilon=bn_epsilon,
                               name='conv_expand_%d_bn' % block_id)(x)
        x = Activation(relu6, name='conv_expand_%d_relu' % block_id)(x)
    else:
        x = inputs

    #x = SeparableConv2D(pointwise_conv_filters, 3, strides=strides, name='conv_dw_%d' % block_id,
    #                    padding='same', depth_multiplier=depth_multiplier)(x)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)

    x = BatchNormalization(axis=channel_axis,
                           momentum=bn_momentum,
                           epsilon=bn_epsilon,
                           name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)

    x = BatchNormalization(axis=channel_axis,
                           momentum=bn_momentum,
                           epsilon=bn_epsilon,
                           name='conv_pw_%d_bn' % block_id)(x)

    if strides == (2, 2):
        return x
    else:
        if input_shape[channel_axis] == pointwise_conv_filters:

            x = add([inputs, x])

    return x


def SubpixelConv2D(input_shape, scale=4, name='subpixel'):
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [
            input_shape[0], input_shape[1] * scale, input_shape[2] * scale,
            int(input_shape[3] / (scale**2))
        ]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)


#---------------------------------------------------------------------------


def MNv2_segment_depth_multiloss_model(inputShape=(416, 416, 3),
                                       alpha=1.0,
                                       expansion_factor=6,
                                       depth_multiplier=1,
                                       lock_backend_weights=True,
                                       CLASSES=19):

    img_in_1x = Input(shape=inputShape)
    mobilenetv2 = MobileNetV2(input_shape=None,
                              alpha=alpha,
                              depth_multiplier=depth_multiplier,
                              include_top=False,
                              weights='imagenet',
                              input_tensor=img_in_1x)
    model_backend = Model(
        inputs=mobilenetv2.input,
        outputs=[
            mobilenetv2.get_layer('block_16_project_BN').output,  #(13,13,320)
            mobilenetv2.get_layer('block_9_add').output,  #(26,26,64)
            mobilenetv2.get_layer('block_5_add').output,  #(52,52,32)
            mobilenetv2.get_layer('block_2_add').output,  #(104,104,24)
            mobilenetv2.get_layer('expanded_conv_project_BN').output
        ])  #(208,208,16)

    if lock_backend_weights:
        for i in range(len(model_backend.layers)):
            model_backend.layers[i].trainable = False

    f13, f26, f52, f104, f208 = model_backend(img_in_1x)

    x = f13
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_1')(x)  # 26, 26, 80
    x = _depthwise_conv_block_v2(x,
                                 64,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=1)
    x = _depthwise_conv_block_v2(x,
                                 64,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=2)
    x = _depthwise_conv_block_v2(x,
                                 64,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=3)
    x = _depthwise_conv_block_v2(x,
                                 64,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=4)

    # x1 = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=5)
    # x2 = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=6)
    depth_out_1 = x
    seg_out_1 = x
    #3 32, 64, 24

    #f26 = _depthwise_conv_block_v2(f26, 64, alpha, expansion_factor, depth_multiplier, block_id=13)
    #f26 = _depthwise_conv_block_v2(f26, 64, alpha, expansion_factor, depth_multiplier, block_id=16)
    x = concatenate([x, f26])  #26, 26, 128
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_2')(x)  #52, 52, 32
    x = _depthwise_conv_block_v2(x,
                                 32,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=7)
    x = _depthwise_conv_block_v2(x,
                                 32,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=8)
    x = _depthwise_conv_block_v2(x,
                                 32,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=9)

    # x1 = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=10)
    # x2 = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=11)
    depth_out_2 = x
    seg_out_2 = x
    #5 64, 128, 16

    #f52 = _depthwise_conv_block_v2(f52, 32, alpha, expansion_factor, depth_multiplier, block_id=14)
    #f52 = _depthwise_conv_block_v2(f52, 32, alpha, expansion_factor, depth_multiplier, block_id=17)
    x = concatenate([x, f52])  #52, 52, 64
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_3')(x)  #104, 104, 16
    x = _depthwise_conv_block_v2(x,
                                 24,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=12)
    x = _depthwise_conv_block_v2(x,
                                 24,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=13)
    x = _depthwise_conv_block_v2(x,
                                 24,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=14)
    # x1 = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, block_id=15)
    # x2 = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, block_id=16)
    depth_out_3 = x
    seg_out_3 = x
    #7 128, 256, 8

    #f104 = _depthwise_conv_block_v2(f104, 24, alpha, expansion_factor, depth_multiplier, block_id=15)
    #f104 = _depthwise_conv_block_v2(f104, 32, alpha, expansion_factor, depth_multiplier, block_id=18)
    x = concatenate([x, f104])  #104,104,48
    x = SubpixelConv2D(x.shape, scale=2, name='subpixel_4')(x)  #208,208,12
    x = _depthwise_conv_block_v2(x,
                                 16,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=17)
    x = _depthwise_conv_block_v2(x,
                                 16,
                                 alpha,
                                 expansion_factor,
                                 depth_multiplier,
                                 block_id=18)
    # x1 = _depthwise_conv_block_v2(x, 16, alpha, expansion_factor, depth_multiplier, block_id=19)
    # x2 = _depthwise_conv_block_v2(x, 16, alpha, expansion_factor, depth_multiplier, block_id=20)
    depth_out_4 = x
    seg_out_4 = x
    #9 256, 512, 8

    depth_pred_16x = Conv2D(1, 1, name='depth_pw_1')(depth_out_1)
    depth_pred_8x = Conv2D(1, 1, name='depth_pw_2')(depth_out_2)
    depth_pred_4x = Conv2D(1, 1, name='depth_pw_3')(depth_out_3)
    depth_pred_2x = Conv2D(1, 1, name='depth_pw_4')(depth_out_4)

    seg_pred_16x = Conv2D(CLASSES, 1, name='seg_pw_1')(seg_out_1)
    seg_pred_8x = Conv2D(CLASSES, 1, name='seg_pw_2')(seg_out_2)
    seg_pred_4x = Conv2D(CLASSES, 1, name='seg_pw_3')(seg_out_3)
    seg_pred_2x = Conv2D(CLASSES, 1, name='seg_pw_4')(seg_out_4)

    seg_pred_16x = Activation('softmax')(seg_pred_16x)
    seg_pred_8x = Activation('softmax')(seg_pred_8x)
    seg_pred_4x = Activation('softmax')(seg_pred_4x)
    seg_pred_2x = Activation('softmax')(seg_pred_2x)

    model = Model(img_in_1x, [
        depth_pred_2x, depth_pred_4x, depth_pred_8x, depth_pred_16x,
        seg_pred_2x, seg_pred_4x, seg_pred_8x, seg_pred_16x
    ])

    # initialize weighting for decoder layers
    for i in range(len(model_backend.layers), len(model.layers)):
        model.layers[i].kernel_initializer = initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None)

    return model
