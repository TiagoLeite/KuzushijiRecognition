import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import albumentations as A
import random
from glob import glob
import pandas as pd
import uuid
import random as rd


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
random.seed(29)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


def augment(aug, image):
    augmented = aug(image=image)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
    return image_aug


def remove_images(images_list):
    rd.shuffle(images_list)
    diff = len(images_list) - 2000  # max 2000 images in folder
    for img_name in images_list[:diff]:
        command = 'rm ' + img_name
        os.system(command)


strong = A.Compose([
    A.ChannelShuffle(p=1),
    A.RGBShift(),
    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_REPLICATE),
    A.Blur(),
    A.GaussNoise(),
    #A.ElasticTransform(),
], p=1)


image_dir = 'box_images'
img_classes_dir = glob(image_dir+'/*')
files_per_dir = list()
total_classes = len(img_classes_dir)
count = 0
for img_classes in img_classes_dir:
    count += 1
    print(img_classes, count, '/', total_classes)
    img_in_folder = glob(img_classes+'/*')
    total_images = len(img_in_folder)
    files_per_dir.append(total_images)
    #if total_images > 2000:
    #    print('Removing...')
    #    remove_images(img_in_folder)
    '''ended = False
    while (total_images < 500) and (not ended):
        for one_img in img_in_folder:
            original_image = cv2.imread(one_img)
            img_aug = augment(strong, original_image)
            aug_name = one_img.split('.')[0]+'_'+uuid.uuid4().hex[:8]+'_aug.jpg'
            cv2.imwrite(aug_name, img_aug)
            total_images += 1
            if total_images > 500:
                ended = True
                break'''

dataframe = pd.DataFrame(data={'name': img_classes_dir, 'count_images': files_per_dir})
dataframe = dataframe.sort_values(by='count_images', ascending=False)
dataframe.to_csv('classes_count.csv', index=False)
print(dataframe)
#print(dataframe['count_images'].value_counts(bins=100, normalize=True))
#print(len(img_classes_dir))
#print(img_classes_dir)


#r = augment_and_show(strong, image)
#cv2.imwrite('lala.jpg', r)





