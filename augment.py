import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import albumentations as A
import random

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def augment_and_show(aug, image, filename=None):
    augmented = aug(image=image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title('Original image')
    ax[1].imshow(image_aug)
    ax[1].set_title('Augmented image')
    plt.show()
    f.tight_layout()
    if filename is not None:
        f.savefig(filename)
    return augmented['image']


# random.seed(42)
image = cv2.imread('box_images/U+4C99/25250223.jpg')
strong = A.Compose([
    A.ChannelShuffle(p=1),
    A.RGBShift(),
    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_REPLICATE),
    A.Blur(),
    A.GaussNoise(),
    #A.ElasticTransform(),
], p=1)

r = augment_and_show(strong, image)

