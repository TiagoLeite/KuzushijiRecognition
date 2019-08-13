import pandas as pd
from keras.models import load_model
import numpy as np
from keras import backend as K
import cv2 as cv
import glob


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


def parse_row(row_string):
    boxes = row_string.split(' ')[:-1]
    boxes = np.reshape(boxes, newshape=[-1, 5])[:, 1:].astype(np.int)
    return boxes

try:
    submission_full = pd.read_csv('detections/submission_full.csv')
except:
    submission_full = pd.read_csv('submission_full.csv')

labels_map = pd.read_csv('labels_map.csv')

model = load_model('saved_class_model_colab.h5', custom_objects={'recall_score': recall_score,
                                                                 'precision_score': precision_score})
print(model.summary())
lines = []
lines_empty = []
images = []
images_empty = []

size = len(submission_full)

for index, row in submission_full.iterrows():

    img = cv.imread('test_images/' + row['image_id'] + '.jpg')

    try:
        boxes = parse_row(row['labels'])
    except:
        images_empty.append(row['image_id'])
        lines_empty.append(row['labels'])
        continue

    print(index, 'of', size, np.shape(boxes))

    images_box_batch = []
    all_boxes_line = ''

    for box in boxes:
        image_box = img[box[1]:box[3], box[0]:box[2]]
        image_box = cv.resize(image_box, (128, 128))
        image_box = image_box / 255.0
        images_box_batch.append(image_box)

    images_box_batch = np.array(images_box_batch)

    pred = model.predict(images_box_batch,
                         verbose=2, batch_size=128)

    pred_max = [np.argmax(pred_i) for pred_i in pred]

    label = [labels_map.loc[labels_map['index'] == pred_max_i, 'name'].iloc[0] for pred_max_i in pred_max]

    k = 0

    for box in boxes:

        all_boxes_line += (
                str(label[k]) + ' ' + str(int((box[0] + box[2]) / 2)) + ' ' + str(int((box[1] + box[3]) / 2)) + ' ')
        k += 1

    lines.append(all_boxes_line)
    images.append(row['image_id'])

lines.extend(lines_empty)
images.extend(images_empty)

re_submission = pd.DataFrame(data={'image_id': images, 'labels': lines})
re_submission.to_csv('re_submission.csv', index=False)


