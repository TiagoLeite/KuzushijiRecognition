import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import *
from keras.models import Model
import cv2 as cv
import argparse
import tensorflow as tf
from keras import backend as K
import pandas as pd
import numpy as np
import glob

train_path = 'box_images/'
BATCH_SIZE = 128
EPOCHS = 1
IMAGE_NEW_SHAPE = (128, 128)


def slice_image(image_source_folder, image_dest_folder, slice_shape=IMAGE_NEW_SHAPE):

    image_count = 0
    all_images = glob.glob(image_source_folder + '/*.jpg')

    for image_name in all_images:

        image_count += 1
        print(image_count, ' of ', len(all_images))

        image = cv.imread(image_name)
        image_shape = np.shape(image)
        image_width = image_shape[1]
        image_height = image_shape[0]
        rec = image.copy()

        n_slices_x = image_width // slice_shape[0]
        # n_slices_x *= 2
        n_slices_x += int(n_slices_x/2.5)

        n_slices_y = image_height // slice_shape[1]
        # n_slices_y *= 2
        n_slices_y += int(n_slices_y/2.5)

        step_x = slice_shape[0] - (slice_shape[0] * (n_slices_x + 1) - image_width) / n_slices_x
        step_y = slice_shape[1] - (slice_shape[1] * (n_slices_y + 1) - image_height) / n_slices_y

        # step_x /= 2
        # step_y /= 2

        count = 0
        index_x, index_y = -1, -1  # will help to name the images

        for x in range(0, image_width - slice_shape[0] + 1, int(step_x)):
            index_x += 1
            index_y = -1
            for y in range(0, image_height - slice_shape[1] + 1, int(step_y)):

                index_y += 1
                rec = cv.rectangle(rec, (x, y), (x + slice_shape[0],
                                                 y + slice_shape[1]), (0, 0, 255), 3)
                count += 1
                roi = image[y:(y + slice_shape[1]), x:(x + slice_shape[0])]

                saved_name = image_dest_folder + '/' + image_name.split('/')[-1].replace(".jpg", '') + '-' \
                             + str(x) + '_' + str(y) + '.jpg'

                cv.imwrite(saved_name, roi)

        #cv.imwrite(image_dest_folder + "/sliced_" + image_name.split('/')[-1], rec)
        #print("image/sliced_" + image_name.split('/')[-1], 'Slices:', count)


def get_new_model():
    mobile = keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                    input_tensor=Input(shape=(128, 128, 3)))
    # print(mobile.summary())
    x = mobile.layers[-4].output
    reshaped = Reshape(target_shape=[1024], name='tiago_reshape')(x)
    # intermediate = Dense(512, activation='relu')(reshaped)
    # drop = Dropout(0.5)(intermediate)
    pred = Dense(CLASSES_NUM, activation='softmax')(reshaped)
    model = Model(inputs=mobile.input, outputs=pred)
    return model


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


FLAGS = None
CLASSES_NUM = 4212

train_datagen = ImageDataGenerator(preprocessing_function=None,
                                   rescale=1.0/255.0,
                                   # rotation_range=180,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   # shear_range=0.2,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   # zoom_range=[0.5, 1.5],
                                   brightness_range=[0.8, 1.2],
                                   validation_split=0.15)

train_gen = train_datagen.flow_from_directory(train_path,
                                              target_size=(128, 128),
                                              batch_size=BATCH_SIZE,
                                              subset='training')

val_gen = train_datagen.flow_from_directory(train_path,
                                            target_size=(128, 128),
                                            batch_size=BATCH_SIZE,
                                            subset='validation')

#if SAVED_MODEL_PATH == 'null':
#    model = get_new_model()
#else:
#    print('Restoring model from ', SAVED_MODEL_PATH)
#    model = load_trained_model(SAVED_MODEL_PATH, True)  # change to False if not training from checkpoint

# model = get_new_model()

model = load_model('saved_class_model.h5',
                   custom_objects = {'recall_score': recall_score,
                                    'precision_score': precision_score})

print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy', recall_score, precision_score])

label_map = train_gen.class_indices
keys = list(label_map.keys())
values = [label_map[key] for key in keys]
print(np.shape(keys), np.shape(values))
dataframe = pd.DataFrame(columns=['name', 'index'], data=np.transpose([keys, values]))
print(dataframe.head())
print(dataframe.tail())
dataframe.to_csv('labels_map.csv', index=False)

model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.samples // BATCH_SIZE,
                    validation_data=val_gen,
                    validation_steps=val_gen.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    workers=-1)

model.save('saved_class_model_2.h5')

# frozen_graph = freeze_session(K.get_session(),
#                              output_names=[out.op.name for out in model.outputs])
# tf.train.write_graph(frozen_graph, logdir='saved_models', name="saved_model_78.pb", as_text=False)
