from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import *
from matplotlib import pyplot as plt
from keras.models import load_model
from random import shuffle
import glob
import cv2 as cv
from keras import backend as K
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


EPOCHS = 10
TARGET_SIZE = (128, 128)
color_mode = 'rgb'
channels = 3


def create_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    # bottleneck layer:
    x = Flatten()(x)
    encoder = Dense(units=512, name='encoder')(x)
    x = Dense(units=8 * 8 * 32)(encoder)
    x = Reshape(target_shape=[8, 8, 32])(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # final(output) layer
    decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder')(x)
    autoencoder = Model(input=input_img, output=decoder)

    # using rmsprop optimizer and binary crossentropy as loss function
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return autoencoder


def train_ae(model):
    train_path = 'box_images/'
    BATCH_SIZE = 32
    train_datagen = ImageDataGenerator(preprocessing_function=None,
                                       rescale=1.0 / 255.0,
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
                                                  target_size=TARGET_SIZE,
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

    filepath = "ckpt_ae/ae_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='min')

    model.fit_generator(train_gen,
                        steps_per_epoch=train_gen.samples // BATCH_SIZE,
                        validation_data=val_gen,
                        validation_steps=val_gen.samples // BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        shuffle=False,
                        callbacks=[checkpoint],
                        workers=-1)

    model.save('saved_models/autoencoder.h5')


def get_encodings(autoencoder):
    encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    train_path = 'box_images/'
    BATCH_SIZE = 64
    train_datagen = ImageDataGenerator(preprocessing_function=None,
                                       rescale=1.0 / 255.0,
                                       # rotation_range=180,
                                       # width_shift_range=0.25,
                                       # height_shift_range=0.25,
                                       # shear_range=0.2,
                                       # horizontal_flip=True,
                                       # vertical_flip=True,
                                       # zoom_range=[0.5, 1.5],
                                       # brightness_range=[0.75, 1.25],
                                       validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(train_path,
                                                  class_mode='categorical',
                                                  color_mode=color_mode,
                                                  target_size=TARGET_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  subset='training')

    val_gen = train_datagen.flow_from_directory(train_path,
                                                class_mode='categorical',
                                                color_mode=color_mode,
                                                shuffle=False,
                                                target_size=TARGET_SIZE,
                                                batch_size=BATCH_SIZE,
                                                subset='validation')

    encoding_train = encoder_model.predict_generator(train_gen, steps=int(np.ceil(train_gen.samples/BATCH_SIZE)),
                                                     workers=-1, verbose=1)
    encoding_test = encoder_model.predict_generator(val_gen, steps=int(np.ceil(val_gen.samples/BATCH_SIZE)),
                                                    workers=-1, verbose=1)
    y_train = train_gen.classes
    y_test = val_gen.classes

    label_map = train_gen.class_indices
    keys = list(label_map.keys())
    values = [label_map[key] for key in keys]
    print(np.shape(keys), np.shape(values))
    dataframe = pd.DataFrame(columns=['name', 'index'], data=np.transpose([keys, values]))
    print(dataframe.head())
    print(dataframe.tail())
    dataframe.to_csv('ae_labels_map.csv', index=False)

    np.savez_compressed('x_train.npz', x_train=encoding_train)
    np.savez_compressed('x_test.npz', x_test=encoding_test)

    np.savez_compressed('y_train.npz', y_train=y_train)
    np.savez_compressed('y_test.npz', y_test=y_test)

    return encoding_train, encoding_test, y_train, y_test


def test_model_plot_image(model, image_test_path):
    img = cv.imread(image_test_path)
    img = cv.resize(img, (128, 128))
    img2 = img / 255.0
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
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[7].output])
    layer_output = get_3rd_layer_output([image_test_path])
    return layer_output[0]


def get_all_embeddings(model):
    images_path = glob.glob('box_images/*/*')
    images_list = []
    count = 0
    images_batch = []
    last_image = ' '

    for image in images_path:

        if image.split('/')[1] == last_image:
            continue

        last_image = image.split('/')[1]
        print('Read ', count, 'of 4224')
        count += 1
        img = cv.imread(image)
        img = cv.resize(img, (128, 128))
        img = img / 255.0
        images_batch.append(img)
        images_list.append(image.split('/')[-2])

    images_batch = np.array(images_batch)
    print('All images:', np.shape(images_batch))
    emb = get_embeddings(model, images_batch)
    np.savez_compressed('embeddings.npz', img_name=images_list, emb=emb)


def knn(emb, all_emb, all_clazz):
    min_dist = np.inf
    clazz = -1
    for one_emb in all_emb:
        this_dist = np.linalg.norm(one_emb - emb)
        if this_dist < min_dist:
            min_dist = this_dist
            clazz = all_emb.index(one_emb)
    return all_clazz[clazz]


# autoencoder_model = create_autoencoder(input_shape=(128, 128, 3))
autoencoder_model = load_model('ckpt_ae/ae_07.h5')
print(autoencoder_model.summary())
#test_model_plot_image(autoencoder_model, 'box_images/U+6991/5f06775d.jpg')

#train_ae(autoencoder_model)
# x_train, x_test, y_train, y_test = get_encodings(autoencoder_model)

x_train = np.load("x_train.npz")['x_train']
x_test = np.load("x_test.npz")['x_test']
y_train = np.load("y_train.npz")['y_train']
y_test = np.load("y_test.npz")['y_test']

print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))
print(np.shape(y_test))

k_nn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
k_nn.fit(x_train, y_train)

print(y_train[-100])

print(k_nn.predict([x_train[-100]]))


# encoded_train = np.asarray(encoder_model.predict())

# =======================================================================
# test_model(autoencoder_model, 'box_images/U+6991/5f06775d.jpg')
# get_all_embeddings(autoencoder_model)
'''data = np.load("embeddings.npz")
print(data['emb'])
print(data['img_name'])
print(np.shape(data['emb']))'''
