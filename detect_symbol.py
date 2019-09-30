import keras
import keras_retinanet
import math
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import re
import imageio
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import glob
import cv2 as cv
import keras.backend as K


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


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


# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)
# print(model.summary())
# load label to names mapping for visualization purposes


labels_to_names = {0: 'symbol'}
IMAGE_NEW_SHAPE = (1024, 1024)
keras.backend.tensorflow_backend.set_session(get_session())
model = models.load_model('saved_models/detection_model.h5')


def compute_iou(box1, box2):
    x1 = int(box1[1])
    y1 = int(box1[2])
    xmax1 = int(box1[3])
    ymax1 = int(box1[4])

    x2 = int(box2[1])
    y2 = int(box2[2])
    xmax2 = int(box2[3])
    ymax2 = int(box2[4])

    assert xmax1 > x1
    assert ymax1 > y1
    assert xmax2 > x2
    assert ymax2 > y2

    new_x = max(x1, x2)
    new_y = max(y1, y2)

    new_xmax = min(xmax1, xmax2)
    new_ymax = min(ymax1, ymax2)

    if (new_xmax > new_x) and (new_ymax > new_y):
        inter_area = (new_xmax - new_x) * (new_ymax - new_y)
        union_area = (xmax1 - x1) * (ymax1 - y1) + (xmax2 - x2) * (ymax2 - y2) - inter_area
        return inter_area / union_area

    return 0

# old
def remove_similar_boxes(boxes_csv_file, new_box_csv_file, iou_threshold=0.25):
    boxes = pd.read_csv(boxes_csv_file).values
    print(pd.np.shape(boxes))
    size = len(boxes)

    for k in range(size):
        if k >= len(boxes):
            break
        ious = [[list(another_box), compute_iou(boxes[k], another_box)] for another_box in boxes[(k + 1):]]
        ious = [x[0] for x in ious if x[1] > iou_threshold]
        if len(ious) == 0:
            continue
        boxes = [x for x in boxes if not (ious == x).all(1).any()]

    print("Final:", np.shape(boxes))
    data = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'class'], data=boxes)
    data.to_csv(new_box_csv_file, index=False)
    return boxes


def draw_boxes_on_image(imagefile, boxes_file, filtered_boxes_file):
    image = read_image_bgr(imagefile)
    draw = image.copy()
    boxes = pd.read_csv(filtered_boxes_file)
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    for index, row in boxes.iterrows():
        draw_box(draw, [int(row['xmin']),
                        int(row['ymin']),
                        int(row['xmax']),
                        int(row['ymax'])],
                 color=(0, 255, 0), thickness=2)
        draw_caption(draw, [int(row['xmin']),
                            int(row['ymin']),
                            int(row['xmax']),
                            int(row['ymax'])], labels_to_names[int(row['class'])])
    draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite('detections/' + str(len(boxes)) + '_boxes_detected_' + imagefile.split('/')[-1], draw_conv)


# Returns absolute boxes already
def run_detection_on_image(image_path, min_score=0.5):
    image = read_image_bgr(image_path)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # correct for image scale
    boxes /= scale
    return_boxes = []
    return_scores = []
    return_image_name = []
    # captions = list()

    lala = re.sub("[A-Z a-z /.]+", "", image_path).split('_')[-2:]

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < min_score:
            break
        b = box.astype(int)
        b = list(b)
        absolute_boxes = [
            b[0],
            b[1],
            b[2],
            b[3], label]
        return_boxes.append(absolute_boxes)
        return_scores.append(score)
        return_image_name.append(image_path.split('/')[-1])
        # draw_box(draw, b, (0, 255, 0))
        # gt = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # draw_box(draw, gt, color=(255, 0, 0))
        # caption = "{} {:.3f}".format(labels_to_names[label], score)
        # captions.append(caption)
        # draw_caption(draw, b, caption)

    # file, ext = os.path.splitext(image_path)
    # image_name = file.split('/')[-1] + ext
    # output_path = os.path.join('results/', image_name)
    # draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(output_path, draw_conv)

    return return_boxes, return_scores, return_image_name


def detect_birds_in_folder(foldername, file_to_save, min_score):
    # test_images = glob.glob(foldername + '/' + prefix[:-4] + '*')
    test_images = glob.glob(foldername + '/*')
    test_images = sorted(test_images)
    # test_images = sorted(test_images)
    boxes = list()
    scores = list()
    images_ids = list()
    count = 0

    for image_name in test_images:

        if count % 10 == 0:
            print("Now detecting: ", count, ' of', len(test_images))

        count += 1
        b, score, img_name = run_detection_on_image(image_name, min_score=min_score)
        boxes.extend(b)
        scores.extend(score)
        images_ids.extend(img_name)

    print(len(images_ids), len(boxes), len(scores))

    dataframe = pd.DataFrame(np.column_stack([images_ids, boxes, scores]),
                             columns=['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score'])

    # dataframe = pd.DataFrame(columns=['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score'],
    #                         data=[images_ids, boxes, scores])

    dataframe.to_csv(file_to_save, index=False)


def get_steps(image_name, slice_shape=IMAGE_NEW_SHAPE):
    image = cv.imread(image_name)
    image_shape = np.shape(image)
    image_width = image_shape[1]
    image_height = image_shape[0]

    n_slices_x = image_width // slice_shape[0]
    n_slices_y = image_height // slice_shape[1]

    step_x = slice_shape[0] - (slice_shape[0] * (n_slices_x + 1) - image_width) / n_slices_x
    step_y = slice_shape[1] - (slice_shape[1] * (n_slices_y + 1) - image_height) / n_slices_y

    return int(step_x), int(step_y)


def slice_image(image_name, slice_shape=IMAGE_NEW_SHAPE):

    image = cv.imread(image_name)
    image_shape = np.shape(image)
    image_width = image_shape[1]
    image_height = image_shape[0]
    # rec = image.copy()
    n_slices_x = image_width // slice_shape[0]
    n_slices_y = image_height // slice_shape[1]

    step_x = slice_shape[0] - (slice_shape[0] * (n_slices_x + 1) - image_width) / n_slices_x
    step_y = slice_shape[1] - (slice_shape[1] * (n_slices_y + 1) - image_height) / n_slices_y

    count = 0
    index_x, index_y = -1, -1  # will help to name the images

    for x in range(0, image_width - slice_shape[0] + 1, int(step_x)):
        index_x += 1
        index_y = -1
        for y in range(0, image_height - slice_shape[1] + 1, int(step_y)):
            index_y += 1
            # rec = cv.rectangle(rec, (x, y), (x + slice_shape[0],
            #                                 y + slice_shape[1]), (0, 0, 255), 3)
            count += 1
            roi = image[y:(y + slice_shape[1]), x:(x + slice_shape[0])]
            saved_name = 'test_sliced_images/' + image_name.split('/')[-1].replace(".jpg", '') + '_' \
                         + str(index_x) + '_' + str(index_y) + '.jpg'
            cv.imwrite(saved_name, roi)

    # cv.imwrite("detections/" + image_name.split('/')[-1], rec)
    return int(step_x), int(step_y)


def recompose_image():

    images = glob.glob('test_images/*')
    data = pd.read_csv('detections/all_detections.csv')
    count = 1
    size = len(images)

    for image in images:
        print(count, 'of', size)
        count += 1

        image_split = image.split('/')[-1]

        data_image = data[data['image_id'].str.startswith(image_split.split('.')[0])]

        if len(data_image) == 0:
            continue

        rec = cv2.imread(image)
        step_x, step_y = get_steps(image)
        image = image.split('/')[-1]
        print('Found')

        for index, row in data_image.iterrows():
            image_inc = row['image_id'].split('.')[0].split('_')[-2:]
            incx = int(image_inc[0])
            incy = int(image_inc[1])
            rec = cv.rectangle(rec, (int(row['xmin']) + step_x * incx,
                                     int(row['ymin']) + step_y * incy),
                               (int(row['xmax']) + step_x * incx,
                                int(row['ymax']) + step_y * incy),
                               (0, 255, 0), 3)
        try:
            cv2.imwrite('detections/debug_detections/' + str(len(data_image)) + '_boxes_' + image, rec)
            print('Saved')
        except:
            print('Error')


images = glob.glob('test_images/*')
lines = []
lines_full = []
size = len(images)
count = 0

classes_model = keras.models.load_model('saved_class_model.h5',
                                        custom_objects={'recall_score': recall_score,
                                                        'precision_score': precision_score})

labels_map = pd.read_csv('labels_map.csv')

for image in images:
    print('Now detecting:', image, count, 'of', size)
    img = cv.imread(image)
    count += 1
    step_x, step_y = get_steps(image)

    detect_birds_in_folder('test_sliced_images', 'detections/all_detections.csv', min_score=0.5)

    # needs to be fixed!
    boxes = remove_similar_boxes('detections/all_detections.csv', 'detections/filtered_detections.csv',
                                 iou_threshold=0.5)

    #draw_boxes_on_image('detections/' + image.split('/')[-1], 'detections/filtered_detections.csv',
    #                    'detections/filtered_detections.csv')
    # cv.imwrite('pico.jpg', images_box[0])

    all_boxes_line = ''
    all_boxes_line_full = ''

    for box in boxes:

        image_box = img[box[1]:box[3], box[0]:box[2]]
        image_box = cv.resize(image_box, (128, 128))
        image_box = image_box/255.0

        pred = classes_model.predict(np.expand_dims(image_box, axis=0),
                                     verbose=0, batch_size=1)[0]

        pred_max = np.argmax(pred)

        label = labels_map.loc[labels_map['index'] == pred_max, 'name'].iloc[0]

        all_boxes_line += (str(label) + ' ' + str(int((box[0] + box[2])/2)) + ' ' + str(int((box[1] + box[3])/2)) + ' ')

        all_boxes_line_full += (str(label) + ' ' + str(box[0]) + ' ' + str(box[1])
                                 + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')



    lines.append(all_boxes_line)
    lines_full.append(all_boxes_line_full)

images = [image.split('/')[-1].replace('.jpg', '') for image in images]

submisison_df = pd.DataFrame(data={'image_id': images, 'labels': lines})
submisison_df.to_csv('detections/submission.csv', index=False)

submisison_full_df = pd.DataFrame(data={'image_id': images, 'labels': lines_full})
submisison_full_df.to_csv('detections/submission_full.csv', index=False)

