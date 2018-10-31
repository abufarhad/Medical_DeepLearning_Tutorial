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
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_1')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_2')(conv2)
    pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2)


    ### Conv 3
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_1')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_2')(conv3)
    pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3)


    ### Conv 4
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_1')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_2')(conv4)
    pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4)


    ### Conv 5
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_1')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_2')(conv5)


    ### upconv + conv 6
    upconv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv6_1')\
              (layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = layers.concatenate([conv4, upconv6], axis=3)

    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv6_2')(merge6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv6_3')(conv6)

    ### Upconv + Conv 7
    upconv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv7_1')(
              layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, upconv7], axis=3)

    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv7_2')(merge7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer='he_normal', name='upconv7_3')(conv7)


    ### Upconv + Conv 8
    upconv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv8_1')\
              (layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, upconv8], axis=3)

    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv8_2')(merge8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv8_3')(conv8)


    ### Upconv + Conv 9
    upconv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name='upconv9_1')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, upconv9], axis=3)

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
    conv2 = layers.Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal', name='conv2_1')(pool1)
    conv2 = layers.BatchNormalization(axis = 3, name= 'conv2_1bn')(conv2)
    conv2 = layers.PReLU(shared_axes=[1, 2], name='prelu2_1')(conv2)

    conv2 = layers.Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal', name='conv2_2')(conv2)
    conv2 = layers.BatchNormalization(axis = 3, name= 'conv2_2bn')(conv2)
    conv2 = layers.PReLU(shared_axes=[1, 2], name='prelu2_2')(conv2)

    pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2)


    ### Conv 3
    conv3 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='conv3_1')(pool2)
    conv3 = layers.BatchNormalization(axis=3, name='conv3_1bn')(conv3)
    conv3 = layers.PReLU(shared_axes=[1, 2], name='prelu3_1')(conv3)

    conv3 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='conv3_2')(conv3)
    conv3 = layers.BatchNormalization(axis=3, name='conv3_2bn')(conv3)
    conv3 = layers.PReLU(shared_axes=[1, 2], name='prelu3_2')(conv3)

    pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3)


    ### Conv 4
    conv4 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='conv4_1')(pool3)
    conv4 = layers.BatchNormalization(axis=3, name='conv4_1bn')(conv4)
    conv4 = layers.PReLU(shared_axes=[1, 2], name='prelu4_1')(conv4)

    conv4 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='conv4_2')(conv4)
    conv4 = layers.BatchNormalization(axis=3, name='conv4_2bn')(conv4)
    conv4 = layers.PReLU(shared_axes=[1, 2], name='prelu4_2')(conv4)

    drop4 = layers.Dropout(0.5)(conv4) ###
    pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(drop4)


    ### Conv 5
    conv5 = layers.Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', name='conv5_1')(pool4)
    conv5 = layers.BatchNormalization(axis=3, name='conv5_1bn')(conv5)
    conv5 = layers.PReLU(shared_axes=[1, 2], name='prelu5_1')(conv5)

    conv5 = layers.Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', name='conv5_2')(conv5)
    conv5 = layers.BatchNormalization(axis=3, name='conv5_2bn')(conv5)
    conv5 = layers.PReLU(shared_axes=[1, 2], name='prelu5_2')(conv5)

    drop5 = layers.Dropout(0.5)(conv5) ###


    ### upconv + conv 6
    upconv6 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv6_1')(layers.UpSampling2D(size=(2, 2))(drop5))
    upconv6 = layers.PReLU(shared_axes=[1, 2], name='prelu6_1')(upconv6)
    merge6 = layers.concatenate([drop4, upconv6], axis=3)

    conv6 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv6_2')(merge6)
    conv6 = layers.PReLU(shared_axes=[1, 2], name='prelu6_2')(conv6)
    conv6 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv6_3')(conv6)
    conv6 = layers.PReLU(shared_axes=[1, 2], name='prelu6_3')(conv6)

    ### Upconv + Conv 7
    upconv7 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv7_1')(layers.UpSampling2D(size=(2, 2))(conv6))
    upconv7 = layers.PReLU(shared_axes=[1, 2], name='prelu7_1')(upconv7)
    merge7 = layers.concatenate([conv3, upconv7], axis=3)

    conv7 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv7_2')(merge7)
    conv7 = layers.PReLU(shared_axes=[1, 2], name='prelu7_2')(conv7)
    conv7 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv7_3')(conv7)
    conv7 = layers.PReLU(shared_axes=[1, 2], name='prelu7_3')(conv7)

    ### Upconv + Conv 8
    upconv8 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv8_1')(layers.UpSampling2D(size=(2, 2))(conv7))
    upconv8 = layers.PReLU(shared_axes=[1, 2], name='prelu8_1')(upconv8)
    merge8 = layers.concatenate([conv2, upconv8], axis=3)

    conv8 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv8_2')(merge8)
    conv8 = layers.PReLU(shared_axes=[1, 2], name='prelu8_2')(conv8)
    conv8 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv8_3')(conv8)
    conv8 = layers.PReLU(shared_axes=[1, 2], name='prelu8_3')(conv8)

    ### Upconv + Conv 9
    upconv9 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='upconv9_1')(layers.UpSampling2D(size=(2, 2))(conv8))
    upconv9 = layers.PReLU(shared_axes=[1, 2], name='prelu9_1')(upconv9)
    merge9 = layers.concatenate([conv1, upconv9], axis=3)

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

