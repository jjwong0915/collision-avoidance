import os
import time
import cv2
import pickle
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from collections import namedtuple
from PIL import Image
from random import shuffle
from keras.preprocessing import image

### Load Data


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def read_image(image_path, shape, rand=-1):
    IMAGE_H, IMAGE_W = shape
    if rand == -1:
        img = image.load_img(image_path, target_size=(IMAGE_H, IMAGE_W))
        img = image.img_to_array(img)
    else:
        img = image.load_img(image_path)
        img = image.img_to_array(img)
        img = cv2.resize(img, (int(IMAGE_H * img.shape[1] / img.shape[0]), IMAGE_H))
        begin = int((img.shape[1] - IMAGE_W) * rand)
        img = img[:, begin:begin + IMAGE_W]
    gamma = random.uniform(0.8, 1.2)
    img = adjust_gamma(np.uint8(img), gamma=gamma)
    img = img.astype(np.float)
    return img


### Result Visualization

Label = namedtuple(
    'Label', 
    [
        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        'category',  # The name of the category that this label belongs to
        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.
        'hasInstances',  # Whether this label distinguishes between single instances or not
        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        'color',  # The color of this label
    ]
)

labels_19 = [
    # Label(name, id, trainId, category, catId, hasInstances, ignoreInEval, color)
    Label('road', 7, 0, 'flat', 1, False, False, (155, 2, 247)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('building', 11, 2, 'construction', 2, False, False, (117, 177, 93)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (253, 88, 255)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (218, 137, 194)),
    Label('sky', 23, 10, 'sky', 5, False, False, (254, 144, 125)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

labels_6 = [
    # Label(name, id, trainId, category, catId, hasInstances, ignoreInEval, color)
    Label('road', 7, 0, 'flat', 1, False, False, (155, 2, 247)),
    Label('sky', 23, 10, 'sky', 5, False, False, (254, 144, 125)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (253, 88, 255)),
    Label('building', 11, 2, 'construction', 2, False, False, (117, 177, 93)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (218, 137, 194)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (253, 88, 255)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (253, 88, 255)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (218, 137, 194)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('building', 11, 2, 'construction', 2, False, False, (117, 177, 93)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('building', 11, 2, 'construction', 2, False, False, (117, 177, 93)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
]

LABEL_NAMES = np.asarray([
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
])


def plot_class_sample(seg_pred, segment_type=19):
    if segment_type == 19:
        labels = labels_19
    elif segment_type == 6:
        labels = labels_6
    seg_pred_vis = np.argmax(seg_pred, axis=3)
    seg_pred_vis = np.squeeze(seg_pred_vis)
    curr_class = []
    for i in range(20):
        if i in seg_pred_vis:
            curr_class.append(i)
    img = np.zeros((int(len(curr_class) / 5) * 100 + 100, 5 * 500, 3), np.uint8)
    img.fill(255)
    counter = 0
    for i in curr_class:
        cv2.rectangle(img, (400 * (counter % 5), 100 * int(counter / 5)),
                      (400 *
                       (counter % 5) + 100, 100 * int(counter / 5) + 100),
                      labels[i].color, -1)
        cv2.putText(img, labels[i].name,
                    (400 *
                     (counter % 5) + 120, int(100 * (int(counter / 5) + 0.6))),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
        counter += 1
    img = cv2.resize(img, (5 * 500 // 1, (int(len(curr_class) / 5) * 100 + 100) // 1))
    return img


def decode_precition(seg_pred, segment_type=19):
    if segment_type == 19:
        labels = labels_19
    elif segment_type == 6:
        labels = labels_6
    seg_argmax = np.argmax(seg_pred, axis=3)
    seg_argmax = np.expand_dims(np.squeeze(seg_argmax), axis=2)
    seg_argmax = np.concatenate((seg_argmax, seg_argmax, seg_argmax), axis=2)
    seg_vis = np.ones((seg_argmax.shape[0], seg_argmax.shape[1], 3))
    ones = np.ones((seg_argmax.shape[0], seg_argmax.shape[1], 3))
    for i in range(19):
        seg_vis = np.where(seg_argmax == [i, i, i],
                           np.array(labels[i].color) * ones, seg_vis)
    return seg_vis


def getResult(model, image, shape):
    INPUT_HEIGHT, INPUT_WIDTH = shape
    # 設定預測深度圖的最小值、最大值
    depth_min, depth_max = 4, 100
    # 建立深度圖和語意分割圖的 variable ，將用來承接 model 的預測結果
    depth = [
        np.zeros((INPUT_HEIGHT // 2, INPUT_WIDTH // 2)),
        np.zeros((INPUT_HEIGHT // 4, INPUT_WIDTH // 4)),
        np.zeros((INPUT_HEIGHT // 8, INPUT_WIDTH // 8)),
        np.zeros((INPUT_HEIGHT // 16, INPUT_WIDTH // 16))
    ]
    segment = [
        np.zeros((INPUT_HEIGHT // 2, INPUT_WIDTH // 2)),
        np.zeros((INPUT_HEIGHT // 4, INPUT_WIDTH // 4)),
        np.zeros((INPUT_HEIGHT // 8, INPUT_WIDTH // 8)),
        np.zeros((INPUT_HEIGHT // 16, INPUT_WIDTH // 16))
    ]
    # model預測結果
    results = model.predict(np.expand_dims(image, axis=0))
    sample = plot_class_sample(results[4])
    # 把每個 resolution 的預測結果，放入 variable: depth 和 segment 中，並進行一些後處理以便顯示
    for i in range(4):
        # 取得該 resolution 的 predicted depth map
        depth[i] = np.squeeze(results[i])
        # 取得該 resolution 的 predicted semantic segmentation
        segment[i] = decode_precition(results[i + 4], segment_type=19)
        # 將預測深度圖以 depth_min, depth_max 進行 clipping
        depth[i][depth[i] > depth_max] = depth_max
        depth[i][depth[i] < depth_min] = depth_min
        # 進行log transform，以便顯示
        depth[i] = (np.log2(depth[i]) - np.log2(depth_min)) * depth_max / (
            np.log2(depth_max) - np.log2(depth_min))
        # 將深度圖進行normalization並inverse，以利顯示
        depth[i] *= 255. / depth_max
        depth[i] = 255 - depth[i]
    return depth, segment


def saveResult(imgs, filename, figsize=None, axis_off=True):
    total_imgs = len(imgs)
    plt.figure(figsize=figsize, dpi=300)
    for i in range(total_imgs):
        plt.subplot(math.ceil(total_imgs/2), 2, i+1)
        plt.imshow(np.uint8(imgs[i]))
        if axis_off:
            plt.axis('off')
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.savefig(filename)
    plt.close()
