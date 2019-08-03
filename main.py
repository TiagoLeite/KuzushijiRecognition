import numpy as np
import cv2 as cv
import glob
import pandas as pd
import re

IMAGE_NEW_SHAPE = (1024, 1024)


def get_boxes_from_string(boxes_string):

    boxes = boxes_string.split(' ')
    boxes = np.reshape(boxes, newshape=[-1, 5])
    # ind = np.argsort(boxes[:, 1])
    # boxes = boxes[ind]
    # ind = np.argsort(boxes[:, 2])
    # boxes = boxes[ind]
    # boxes = [list(box) for box in boxes]
    return boxes


def insert_boxes_into_slices(slices_dir, train_csv):
    data = pd.read_csv(train_csv)
    retina_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    labels_set = set()

    total_rows = np.shape(data)[0]
    print("Total ", total_rows)
    count = 0

    all_slice_images = glob.glob(slices_dir + '/*')

    for index, row in data.iterrows():

        count += 1
        print(count, ' of ', total_rows)

        boxes_string = row['labels']
        image_name = row['image_id']

        if pd.isna(boxes_string):
            continue

        boxes_list = get_boxes_from_string(boxes_string)

        #print(np.shape(boxes_list))
        #print(boxes_list[:5])
        #boxes_list.remove(boxes_list[1])
        #print(boxes_list[:5])
        #input()

        # candidate_images = glob.glob(slices_dir + '/' + image_name + '*')
        regex = re.compile(slices_dir+'/'+image_name + '+')
        candidate_images = list(filter(regex.search, all_slice_images))
        print(candidate_images)

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
                    # print(candidate, box)
                    # np.delete(boxes_list, boxes_list.index(box), axis=0)
                    retina_df = retina_df.append({'filename': candidate_path,
                                                  'xmin': box_x - cand_x,
                                                  'ymin': box_y - cand_y,
                                                  'xmax': box_x + box_w - cand_x,
                                                  'ymax': box_y + box_h - cand_y,
                                                  'class': box_label}, ignore_index=True)

    retina_df.to_csv('retina_train.csv', index=False)
    labels_df = pd.DataFrame(data={'label': list(labels_set), 'id': np.arange(len(labels_set))})
    labels_df.to_csv('labels.csv', index=False)


def slice_image(image_source_folder, image_dest_folder, slice_shape=IMAGE_NEW_SHAPE):
    for image_name in glob.glob(image_source_folder + '/*.jpg'):

        image = cv.imread(image_name)
        image_shape = np.shape(image)
        print('Image Shape:', image_shape)
        image_width = image_shape[1]
        image_height = image_shape[0]
        rec = image.copy()

        reminder_x = image_width % slice_shape[0]
        reminder_y = image_height % slice_shape[1]

        n_slices_x = image_width // slice_shape[0]
        # n_slices_x *= 2
        n_slices_x += int(n_slices_x / 3)

        n_slices_y = image_height // slice_shape[1]
        # n_slices_y *= 2
        n_slices_y += int(n_slices_y / 3)

        print('Rx', reminder_x, 'Ry', reminder_y)

        step_x = slice_shape[0] - (slice_shape[0] * (n_slices_x + 1) - image_width) / n_slices_x
        step_y = slice_shape[1] - (slice_shape[1] * (n_slices_y + 1) - image_height) / n_slices_y

        # step_x /= 2
        # step_y /= 2

        print('Step x:', int(step_x), 'Step y:', int(step_y))

        count = 0
        index_x, index_y = -1, -1  # will help to name the images

        for x in range(0, image_width - slice_shape[0] + 1, int(step_x)):
            index_x += 1
            index_y = -1
            for y in range(0, image_height - slice_shape[1] + 1, int(step_y)):
                print('Step x:', int(step_x), 'Step y:', int(step_y))
                index_y += 1
                rec = cv.rectangle(rec, (x, y), (x + slice_shape[0],
                                                 y + slice_shape[1]), (0, 0, 255), 3)
                count += 1
                roi = image[y:(y + slice_shape[1]), x:(x + slice_shape[0])]

                saved_name = image_dest_folder + '/' + image_name.split('/')[-1].replace(".jpg", '') + '-' \
                             + str(x) + '_' + str(y) + '.jpg'

                cv.imwrite(saved_name, roi)
                print('Saved:', saved_name)

        cv.imwrite(image_dest_folder + "/sliced_" + image_name.split('/')[-1], rec)
        print("image/sliced_" + image_name.split('/')[-1], 'Slices:', count)


# slice_image('mock', 'mock', IMAGE_NEW_SHAPE)
insert_boxes_into_slices('mock', 'train.csv')