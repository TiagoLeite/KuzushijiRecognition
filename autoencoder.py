from keras.layers import *
from keras.losses import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.losses import *
from keras.optimizers import *
from keras_retinanet.losses import focal
from keras.datasets import mnist, cifar10
from matplotlib import pyplot as plt
from keras.models import load_model
from random import shuffle


TARGET_SIZE = (128, 128)
color_mode = 'rgb'
channels = 3

input_layer = Input(shape=[TARGET_SIZE[0], TARGET_SIZE[1], channels])
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(input_layer)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)

x = Flatten()(x)

x = Dense(units=128, name='bottleneck')(x)

x = Dense(units=2048)(x)

x = Reshape(target_shape=(4, 4, -1))(x)

x = Conv2D(         filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

x = Conv2D(         filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

x = Conv2D(         filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

x = Conv2D(         filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

x = Conv2D(         filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
output = Conv2D(channels, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

model = Model(input_layer, output)

model.compile(optimizer=Adam(), loss='binary_crossentropy')

print(model.summary())

train_path = 'box_images/'

BATCH_SIZE = 128

train_datagen = ImageDataGenerator(preprocessing_function=None,
                                   rescale=1.0/255.0,
                                   # rotation_range=180,
                                   # width_shift_range=0.25,
                                   # height_shift_range=0.25,
                                   # shear_range=0.2,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   # zoom_range=[0.5, 1.5],
                                   # brightness_range=[0.6, 1.4],
                                   validation_split=0.1)

train_gen = train_datagen.flow_from_directory(train_path,
                                              class_mode='input',
                                              color_mode=color_mode,
                                              target_size= TARGET_SIZE,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              subset='training')

val_gen = train_datagen.flow_from_directory(train_path,
                                            class_mode='input',
                                            color_mode=color_mode,
                                            shuffle=False,
                                            target_size=TARGET_SIZE,
                                            batch_size=BATCH_SIZE,
                                            subset='validation')

model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.samples // BATCH_SIZE,
                    validation_data=val_gen,
                    validation_steps=val_gen.samples // BATCH_SIZE,
                    epochs=2,
                    verbose=1,
                    shuffle=False,
                    workers=-1)

model.save('autoencoder.h5')
