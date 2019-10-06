import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import *
from keras.models import Model
import cv2 as cv
import argparse
from keras import backend as K
import pandas as pd
import numpy as np
import glob
from keras.callbacks import *
from collections import Counter
from sklearn.utils import class_weight

train_path = 'box_images/'
BATCH_SIZE = 256
EPOCHS = 24
FLAGS = None
CLASSES_NUM = 4212


# IMAGE_NEW_SHAPE = (128, 128)


def get_densenet_model():
    densenet = keras.applications.densenet.DenseNet121(weights='imagenet',
                                                       input_tensor=Input(shape=(128, 128, 3)))
    x = densenet.layers[-2].output
    # x = Conv2D(filters=1024, strides=(2, 2), kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Flatten()(x)
    # x = Reshape(target_shape=[1024], name='reshape')(x)
    # intermediate = Dense(512, activation='relu')(reshaped)
    # drop = Dropout(0.5)(intermediate)
    pred = Dense(CLASSES_NUM, activation='softmax')(x)
    model = Model(inputs=densenet.input, outputs=pred)
    return model


def get_resnet_model():
    densenet = keras.applications.resnet50.ResNet50(weights='imagenet',
                                                    input_tensor=Input(shape=(128, 128, 3)))
    x = densenet.output
    # x = Conv2D(filters=1024, strides=(2, 2), kernel_size=(3, 3), activation='relu', padding='same')(x)
    # x = Flatten()(x)
    x = Reshape(target_shape=[1024], name='reshape')(x)
    # intermediate = Dense(512, activation='relu')(reshaped)
    # drop = Dropout(0.5)(intermediate)
    pred = Dense(CLASSES_NUM, activation='softmax')(x)
    model = Model(inputs=densenet.input, outputs=pred)
    return model


def get_mobilenet_model():
    mobile = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                         input_tensor=Input(shape=(128, 128, 3)))
    print(mobile.summary())
    x = mobile.layers[-2].output # -4 se for mobilenet v1
    # reshaped = Reshape(target_shape=[1280], name='tiago_reshape')(x)
    pred = Dense(CLASSES_NUM, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=pred)
    return model


def get_custom_model():
    input_layer = Input(shape=[128, 128, 3])
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    output = Dense(units=CLASSES_NUM, activation='softmax')(x)
    return Model(input=input_layer, output=output)


def precision_score(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_score(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# model = load_model('saved_models/mobilenet_128.h5',
#                   custom_objects={'recall_score': recall_score,
#                                   'precision_score': precision_score})
#                                   #'focal_loss_fixed': focal_loss(alpha=.25, gamma=2)})

model = get_mobilenet_model()
print(model.summary())

# class_weights = class_weight.compute_class_weight(
#    'balanced',
#    np.unique(train_gen.classes),
#    train_gen.classes)
# print(class_weights)

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy', recall_score, precision_score])

train_datagen = ImageDataGenerator(preprocessing_function=None,
                                   rescale=1.0 / 255.0,
                                   # rotation_range=180,
                                   # width_shift_range=0.1,
                                   # height_shift_range=0.1,
                                   # shear_range=0.1,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   validation_split=0.1)
# zoom_range=[0.8, 1.2],
# brightness_range=[0.75, 1.25])

train_gen = train_datagen.flow_from_directory(train_path,
                                              target_size=(128, 128),
                                              batch_size=BATCH_SIZE,
                                              subset='training')

val_gen = train_datagen.flow_from_directory(train_path,
                                            target_size=(128, 128),
                                            batch_size=BATCH_SIZE,
                                            subset='validation')

label_map = train_gen.class_indices
keys = list(label_map.keys())
values = [label_map[key] for key in keys]
print(np.shape(keys), np.shape(values))
dataframe = pd.DataFrame(columns=['name', 'index'], data=np.transpose([keys, values]))
print(dataframe.head())
print(dataframe.tail())
dataframe.to_csv('labels_map.csv', index=False)

# filepath = "ckpt/mobilenet-{epoch:02d}-{val_acc:.4f}.h5"
filepath = "ckpt/mobilenet128_{epoch:02d}.h5"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=False,
    mode='max')

model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.samples // BATCH_SIZE + 1,
                    validation_data=val_gen,
                    validation_steps=val_gen.samples // BATCH_SIZE + 1,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[checkpoint],
                    # class_weight=class_weights,
                    workers=-1)

model.save('saved_models/mobilenet_final.h5')
