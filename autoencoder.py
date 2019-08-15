from keras.layers import *
from keras.losses import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.losses import *
from keras.optimizers import *
from keras.datasets import mnist, cifar10
from matplotlib import pyplot as plt
from keras.models import load_model
from random import shuffle
from keras.applications import MobileNet
import glob
import cv2 as cv
from keras import backend as K
import pandas as pd

TARGET_SIZE = (128, 128)
color_mode = 'rgb'
channels = 3


def get_mobilenet_ae():

    mobilenet = MobileNet(weights='imagenet', input_tensor=Input(shape=(128, 128, 3)))
    x = mobilenet.layers[-6].output

    x = Dense(1024, name='bottleneck')(x)

    x = Reshape(target_shape=(4, 4, -1))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    output = Conv2D(channels, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input=mobilenet.input, output=output)
    return model


def get_new_model():

    input_layer = Input(shape=[TARGET_SIZE[0], TARGET_SIZE[1], channels])
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(input_layer)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Flatten()(x)

    x = Dense(units=1024, name='bottleneck')(x)

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

    return Model(input_layer, output)


def train_ae(model):
    # model = get_new_model()
    # model = load_model('autoencoder_jap_2.h5')
    train_path = 'box_images/'
    BATCH_SIZE = 4

    train_datagen = ImageDataGenerator(preprocessing_function=None,
                                       rescale=1.0/255.0,
                                       rotation_range=180,
                                       # width_shift_range=0.25,
                                       # height_shift_range=0.25,
                                       # shear_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       # zoom_range=[0.5, 1.5],
                                       brightness_range=[0.75, 1.25],
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
                        epochs=1,
                        verbose=1,
                        shuffle=False,
                        workers=-1)

    model.save('autoencoder_japanese.h5')


def test_model(model, image_test_path):
    img = cv.imread(image_test_path)
    img = cv.resize(img, (128, 128))
    img2 = img/255.0
    img2 = np.expand_dims(img2, axis=0)
    print(np.shape(img2))
    pred = model.predict(img2)[0]
    print(np.shape(pred))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.show()


def get_embeddings(model, image_test_path):
    img = cv.imread(image_test_path)
    img = cv.resize(img, (128, 128))
    img2 = img / 255.0
    img2 = np.expand_dims(img2, axis=0)

    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[7].output])
    layer_output = get_3rd_layer_output([img2])[0]
    return layer_output[0]


def get_all_embeddings(model):
    images_path = glob.glob('box_images/*/*')
    images_list = []
    embeddings = []
    size = len(images_path)
    count = 0
    for image in images_path:
        print(count, 'out of', size)
        count += 1
        images_list.append(image.split('/')[-1])
        emb = get_embeddings(model, image)
        embeddings.append(emb)

    np.savez_compressed('embeddings.npz', img_name=images_list, emb=embeddings)


# model = get_mobilenet_ae()
autoencoder_model = load_model('autoencoder_japanese.h5')
# autoencoder_model.compile(optimizer=Adam(), loss='binary_crossentropy')
print(autoencoder_model.summary())

#test_model(autoencoder_model, 'box_images/U+6991/5f06775d.jpg')
get_all_embeddings(autoencoder_model)
#data = np.load("embeddings.npz")
#print(data['img_name'])
#print(np.shape(data['emb']))




