import pandas as pd
from keras.models import load_model
import numpy as np
from keras import backend as K
import cv2 as cv
import glob
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


def get_embeddings(model, image_test_path):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[7].output])
    layer_output = get_3rd_layer_output([image_test_path])
    return layer_output[0]


def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


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


def get_embeddings(model, image_batch):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[6].output])
    layer_output = get_3rd_layer_output([image_batch])
    return layer_output[0]


def knn(emb, all_emb, all_clazz):
    dists = [np.linalg.norm(one_emb - emb) for one_emb in all_emb]
    min_dist = np.argmin(dists)
    return all_clazz[min_dist]


try:
    submission_full = pd.read_csv('detections/submission_full.csv')
except:
    submission_full = pd.read_csv('submission_full.csv')

labels_map = pd.read_csv('labels_map.csv')
model = load_model('ckpt/mobilenet128_03.h5',
                   custom_objects={'recall_score': recall_score,
                                   'precision_score': precision_score})
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

    all_boxes_line = ''
    images_box_list = []

    for box in boxes:
        image_box = img[box[1]:box[3], box[0]:box[2]]
        image_box = cv.resize(image_box, (128, 128))
        image_box = image_box / 255.0
        # image_box = np.expand_dims(image_box, axis=0)
        images_box_list.append(image_box)

    preds = model.predict(np.array(images_box_list))

    preds_max = [np.argmax(pred) for pred in preds]
    labels = [labels_map.loc[labels_map['index'] == pred_max, 'name'].iloc[0] for pred_max in preds_max]
    k = 0
    for box in boxes:
        all_boxes_line += (
                str(str(labels[k]).split('.')[0]) + ' ' + str(int((box[0] + box[2]) / 2)) + ' ' +
                str(int((box[1] + box[3]) / 2)) + ' ')
        k += 1

    lines.append(all_boxes_line)
    images.append(row['image_id'])

lines.extend(lines_empty)
images.extend(images_empty)
re_submission = pd.DataFrame(data={'image_id': images, 'labels': lines})
re_submission.to_csv('reclass_submission.csv', index=False)
