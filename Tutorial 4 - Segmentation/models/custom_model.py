from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import keras.layers as layers
import keras.models as models

""" Task1 : U-Net """
def UNet_like(input_tensor = None):

    img_input = input_tensor

    ### Conv1
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_1')(img_input)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_2')(conv1)
    pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1)


    ### Conv 2
    pass


    ### Conv 3
    pass


    ### Conv 4
    pass


    ### Conv 5
    pass


    ### upconv + conv 6
    upconv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv6_1')\
              (layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.concatenate([conv4, upconv6], axis=3)

    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv6_2')(merge6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv6_3')(conv6)

    ### Upconv + Conv 7
    pass

    ### Upconv + Conv 8
    pass

    ### Upconv + Conv 9
    pass

    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv9_2')(merge9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv9_3')(conv9)
    conv9 = layers.Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv9_4')(conv9)

    ### Conv 10
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid', name = 'conv10')(conv9)


    # Create model
    model = models.Model(img_input, conv10, name='U-Net')


    return model



""" Task2 : U-Net + Batch Normalization + PReLU + Dropout """
def UNet_like2(input_tensor = None):

    img_input = input_tensor

    ### Conv1
    conv1 = layers.Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal', name='conv1_1')(img_input)
    conv1 = layers.BatchNormalization(axis = 3, name= 'conv1_1bn')(conv1)
    conv1 = layers.PReLU(shared_axes=[1, 2], name='prelu1_1')(conv1)

    conv1 = layers.Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal', name='conv1_2')(conv1)
    conv1 = layers.BatchNormalization(axis = 3, name= 'conv1_2bn')(conv1)
    conv1 = layers.PReLU(shared_axes=[1, 2], name='prelu1_2')(conv1)

    pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1)


    ### Conv 2
    pass


    ### Conv 3
    pass


    ### Conv 4
    pass


    ### Conv 5
    pass


    ### upconv + conv 6
    upconv6 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv6_1')(layers.UpSampling2D(size=(2, 2))(drop5))
    upconv6 = layers.PReLU(shared_axes=[1, 2], name='prelu6_1')(upconv6)
    merge6 = layers.concatenate([drop4, upconv6], axis=3)

    conv6 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv6_2')(merge6)
    conv6 = layers.PReLU(shared_axes=[1, 2], name='prelu6_2')(conv6)
    conv6 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv6_3')(conv6)
    conv6 = layers.PReLU(shared_axes=[1, 2], name='prelu6_3')(conv6)

    ### Upconv + Conv 7
    pass


    ### Upconv + Conv 8
    pass


    ### Upconv + Conv 9
    pass

    conv9 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv9_2')(merge9)
    conv9 = layers.PReLU(shared_axes=[1, 2], name='prelu9_2')(conv9)
    conv9 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv9_3')(conv9)
    conv9 = layers.PReLU(shared_axes=[1, 2], name='prelu9_3')(conv9)
    conv9 = layers.Conv2D(2, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv9_4')(conv9)
    conv9 = layers.PReLU(shared_axes=[1, 2], name='prelu9_4')(conv9)

    ### Conv 10
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid', name = 'conv10')(conv9)


    # Create model
    model = models.Model(img_input, conv10, name='U-Net')


    return model

