from keras.layers import Input, Conv3D, BatchNormalization, ReLU, add, MaxPooling3D, Lambda, \
                         GlobalAveragePooling3D, Dense
from keras.models import Model
import keras.backend as K
import tensorflow as tf


n_blocks = {18: [2,2,2,2], 34: [3,4,6,3]}
n_filters = [64, 128, 256, 512]


def resnet_3d(input_shape=(5,256,256,1), n_classes=10, depth=18):

    inpt = Input(input_shape)

    # conv1-stem: 7x7x7 stride1x2x2
    x = ConvBN(inpt, 64, 7, strides=(1,2,2), padding='same', activation='relu')
    x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)

    num_blocks = n_blocks[depth]
    # conv2
    x = resBlock(x, n_filters[0], strides=(1,2,2), n_blocks=num_blocks[0])

    # conv3
    x = resBlock(x, n_filters[1], strides=(1,2,2), n_blocks=num_blocks[1])

    # conv4
    x = resBlock(x, n_filters[2], strides=(2,2,2), n_blocks=num_blocks[2])

    # conv5
    x = resBlock(x, n_filters[3], strides=(2,2,2), n_blocks=num_blocks[3])

    # head
    x = GlobalAveragePooling3D()(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)

    # model
    model = Model(inpt, x)

    return model


def resBlock(inpt, filters, kernel_size=3, strides=1, n_blocks=2):
    # residual
    x = ConvBN(inpt, filters, kernel_size, strides=strides, padding='same', activation='relu')
    x = ConvBN(x, filters, kernel_size, strides=1, padding='same', activation=None)

    # id path: conv1, zero-padding
    if 2 in strides:
        inpt = ConvBN(inpt, filters, kernel_size=1, strides=strides, padding='same', activation='relu')
    if filters>K.int_shape(inpt)[-1]:
        gap = filters - K.int_shape(inpt)[-1]
        inpt = Lambda(tf.pad, arguments={'paddings': [[0,0],[0,0],[0,0],[0,0],[0,gap]]})(inpt)

    # add
    x = add([inpt, x])
    return x


def ConvBN(x, filters, kernel_size, strides=1, padding='same', activation=None):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = resnet_3d(input_shape=(5,256,256,1), depth=18)
    model.summary()



