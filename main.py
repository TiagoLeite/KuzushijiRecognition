import numpy as np
import cv2 as cv
import glob
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import uuid


IMAGE_NEW_SHAPE = (1024, 1024)


def get_boxes_from_string(boxes_string):

    boxes = boxes_string.split(' ')
    boxes = np.reshape(boxes, newshape=[-1, 5])
    return boxes


def insert_boxes_into_slices(slices_dir, train_csv):
    data = pd.read_csv(train_csv)
    # retina_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    labels_set = set()
    total_rows = np.shape(data)[0]
    print("Total ", total_rows)
    count = 0

    filenames = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    clazzes = []

    for index, row in data.iterrows():

        count += 1
        print(count, ' of ', total_rows)

        boxes_string = row['labels']
        image_name = row['image_id']

        if pd.isna(boxes_string):
            continue

        boxes_list = get_boxes_from_string(boxes_string)

        candidate_images = glob.glob(slices_dir + '/' + image_name + '*')

        for candidate_path in candidate_images:

            candidate = re.split('/|-', candidate_path)

            candidate = candidate[-1].split('.')[0].split('_')
            cand_x = int(candidate[0])
            cand_y = int(candidate[1])

            for box in boxes_list:

                box_label = box[0]
                labels_set.add(box_label)
                box_x = int(box[1])
                box_y = int(box[2])
                box_w = int(box[3])
                box_h = int(box[4])

                if (box_x >= cand_x) and (box_x + box_w <= cand_x + 1024) \
                        and (box_y >= cand_y) and (box_y + box_h <= cand_y + 1024):

                    filenames.append(candidate_path)
                    xmin.append(box_x - cand_x)
                    ymin.append(box_y - cand_y)
                    xmax.append(box_x + box_w - cand_x)
                    ymax.append(box_y + box_h - cand_y)
                    clazzes.append(box_label)

    retina_df = pd.DataFrame(data={'filename': filenames,
                                   'xmin': xmin,
                                   'ymin': ymin,
                                   'xmax': xmax,
                                   'ymax': ymax,
                                   'class': clazzes})

    columnsTitles = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    retina_df = retina_df.reindex(columns=columnsTitles)
    retina_df['class'] = 'symbol'
    retina_df.to_csv('retina_train_no_labels.csv', index=False)
    labels_df = pd.DataFrame(data={'label': ['symbol'], 'id': [0]})
    labels_df = labels_df.reindex(columns=['label', 'id'])
    labels_df['label'] = 'symbol'
    labels_df.to_csv('labels.csv', index=False, header=False)


def save_boxes_to_images(dest_folder_name, original_images_folder):

    train_df = pd.read_csv('train.csv')
    total = len(train_df)

    for index, row in train_df.iterrows():

        sub_folder_name = row['image_id']
        boxes_string = row['labels']

        if pd.isna(boxes_string):
            continue

        boxes_list = get_boxes_from_string(boxes_string)

        print(index, ' of ', total, len(boxes_list), 'boxes')

        image = cv.imread(original_images_folder + '/' + sub_folder_name + '.jpg')

        for box in boxes_list:

            box_label = box[0]
            box_x = int(box[1])
            box_y = int(box[2])
            box_w = int(box[3])
            box_h = int(box[4])

            if not os.path.isdir(dest_folder_name + '/' + box_label):
                os.system('mkdir ' + dest_folder_name + '/' + box_label)

            roi = image[box_y:(box_y + box_h), box_x:(box_x + box_w)]

            saved_name = dest_folder_name + '/' + box_label + '/' + uuid.uuid4().hex[:8] + '.jpg'

            cv.imwrite(saved_name, roi)


def slice_image(image_source_folder, image_dest_folder, slice_shape=IMAGE_NEW_SHAPE):

    image_count = 0
    all_images = glob.glob(image_source_folder + '/*.jpg')

    for image_name in all_images:

        image_count += 1
        if image_count % 10 == 0:
            print(image_count, ' of ', len(all_images))

        image = cv.imread(image_name)
        image_shape = np.shape(image)
        image_width = image_shape[1]
        image_height = image_shape[0]
        rec = image.copy()

        n_slices_x = image_width // slice_shape[0]
        n_slices_x += int(n_slices_x/2.5)

        n_slices_y = image_height // slice_shape[1]
        n_slices_y += int(n_slices_y/2.5)

        step_x = slice_shape[0] - (slice_shape[0] * (n_slices_x + 1) - image_width) / n_slices_x
        step_y = slice_shape[1] - (slice_shape[1] * (n_slices_y + 1) - image_height) / n_slices_y

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

        # cv.imwrite(image_dest_folder + "/sliced_" + image_name.split('/')[-1], rec)


def split_train_test(filename):
    data = pd.read_csv(filename)
    unique_names = data['filename'].unique()
    train, test = train_test_split(unique_names, test_size=0.01, random_state=5)
    data_test = data[data['filename'].isin(test)]
    print('Test:', np.shape(data_test))
    data_train = data[data['filename'].isin(train)]
    print('Train:', np.shape(data_train))
    data_test.to_csv('test_dataset.csv', index=False, header=False)
    data_train.to_csv('train_dataset.csv', index=False, header=False)


slice_image('train_images', 'mock', IMAGE_NEW_SHAPE)
insert_boxes_into_slices('mock', 'train.csv')
split_train_test('retina_train_no_labels.csv')

#save_boxes_to_images('box_images', 'train_images')




