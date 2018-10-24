from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from . import get_submodules_from_kwargs

import keras.layers as layers
import keras.models as models

""" Task1 : VGG 16 """
""" Task4 : VGG16 + Inception Module (VGG block 2~3 사이) + Residual Block (VGG block 3~4 사이) """
def VGG16_residual_inception(include_top=True, input_tensor=None, pooling=None, classes=None):

    img_input = input_tensor

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    pass

    x = layers.MaxPooling2D((2, 2),
                            strides=(2, 2),
                            name='block1_pool')(x)

    # Block 2
    pass


    """ Inception Module """
    ###
    pass
    ###


    # Block 3
    pass


    """ Residual Block """
    ###
    pass
    ###


    # Block 4
    pass

    # Block 5
    pass

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)

        pass

        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Create model.
    model = models.Model(img_input, x, name='vgg16')

    return model


""" Task 2 : Residual Block """
def skip_conncect(input, filters, kernel_size, channel):
    # Block a
    x = layers.Conv2D(filters, kernel_size,
                      padding='same',
                      name='res_1a')(input)
    x = layers.BatchNormalization(axis=channel, name='bn_1b')(x)
    x = layers.Activation('relu')(x)

    # Block b
    x = layers.Conv2D(filters, kernel_size,
                      padding='same',
                      name= 'res_2b')(x)
    x = layers.BatchNormalization(axis=channel, name='bn_2b')(x)

    # Add
    pass
    x = layers.Activation('relu')(x)
    return x


""" Task 3 : Inception Module """
def conv2d_bn(x, filters, num_row, num_col, channel,
              conv_name, bn_name, name=None):

    x = layers.Conv2D(filters, (num_row, num_col),
                      padding='same',
                      name=conv_name)(x)
    x = layers.BatchNormalization(axis=channel, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)

    return x


def inception(x, filters, channel):
    # Branch a
    branch1x1a = conv2d_bn(x, filters, 1, 1, channel, 'branch1x1a_conv', 'branch1x1a_bn')

    # Branch b
    pass

    # Branch c
    pass

    # Branch d
    branch_maxpool5x5d = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='branch5x5d_maxpool')(x)
    branch1x1d = conv2d_bn(branch_maxpool5x5d, filters, 1, 1, channel,'branch1x1d_conv', 'branch1x1d_bn')


    branch_out = layers.concatenate([branch1x1a, branch3x3b, branch5x5c, branch1x1d], axis=channel,name='mixed0')
    return branch_out



