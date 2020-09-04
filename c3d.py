from keras.layers import Input, Conv3D, MaxPooling3D, ReLU, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model


n_blocks = {1,1,2,2,2}
n_filters = [64, 128, 256, 512, 512, 4096, 4096]


def c3d(input_shape=(16,160,160,3), n_classes=10):
    inpt = Input(input_shape)

    # conv1
    x = ConvBN(inpt, n_filters[0], kernel_size=3, strides=1, padding='same', activation='relu')
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1,2,2), padding='same')(x)

    # conv2
    x = ConvBN(x, n_filters[1], kernel_size=3, strides=1, padding='same', activation='relu')
    x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)

    # conv3
    x = ConvBN(x, n_filters[2], kernel_size=3, strides=1, padding='same', activation='relu')
    x = ConvBN(x, n_filters[2], kernel_size=3, strides=1, padding='same', activation='relu')
    x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)

    # conv4
    x = ConvBN(x, n_filters[3], kernel_size=3, strides=1, padding='same', activation='relu')
    x = ConvBN(x, n_filters[3], kernel_size=3, strides=1, padding='same', activation='relu')
    x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)

    # conv5
    x = ConvBN(x, n_filters[4], kernel_size=3, strides=1, padding='same', activation='relu')
    x = ConvBN(x, n_filters[4], kernel_size=3, strides=1, padding='same', activation='relu')
    x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)

    # fc6
    x = Flatten()(x)
    x = Dropout(0.9)(x)
    x = Dense(n_filters[5])(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # fc7
    x = Dropout(0.9)(x)
    x = Dense(n_filters[6])(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # head
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inpt, x)

    return model


def ConvBN(x, filters, kernel_size, strides=1, padding='same', activation=None):
    x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    model = c3d(input_shape=(16,160,160,3), n_classes=10)
    model.summary()




