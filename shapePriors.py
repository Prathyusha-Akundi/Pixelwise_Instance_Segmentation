import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
import cv2


# Get all the bounding boxes and related information
def bboxData(img_dict):
    bbox_img_name = []
    bbox_cls = []
    bbox = []

    for img_name in img_dict:
        img_bbox = img_dict[img_name]["bbox"]
        for box in img_bbox:
            bbox_img_name.append(img_name)
            # Consider box = [x1, y1, x2, y2, class]
            bbox_cls.append(box[4])
            bbox.append(box[0:3])
    return bbox_img_name, bbox_cls, bbox


# Pool bbox with aspect ratios
def bboxARPool(bbox):

    bbox = np.array(bbox)
    w_h = np.array([bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]])
    bbox_ar = w_h[:, 0] / w_h[:, 1]

    return bbox_ar

# Classify aspect ratios in 5 K-Means clusters
def clusterAR(bbox_ar, num_clusters):
    ar_model = KMeans(n_clusters = num_clusters).fit(bbox_ar)
    ar_labels = ar_model.labels_
    return ar_labels


# Extract binary mask and resize to thumbnails

# thumb_size is a dictionary containing thumnail size information related to a given aspect ratio
# thumb_size = {"1":[r,c],"2":[r,c]}
# Or it could be thumb_size = 50
def softMaskDict(bbox, bbox_img_name, bbox_cls, bbox_ar, ar_labels, thumb_size):
    bbox_dict = {}

    for bbox_idx in range(0,len(bbox)):
        box = bbox[bbox_idx]
        img_name = bbox_img_name[bbox_idx]
        cls = bbox_cls[bbox_idx]
        lbl = ar_labels[bbox_idx]
        ar = bbox_ar[bbox_idx]

        if ar < 1:
            size = (thumb_size, thumb_size. / ar)
        else:
            size = (thumb_size * ar, thumb_size)

        img = cv2.imread(img_name)
        crop_img = img[box[1]:box[3], box[0]:box[2]]
        # thumb_img = cv2.resize(crop_img, size)
        # thumb_b, thumb_g, thumb_r = cv2.split(thumb_img)
        thumb_img = np.concatenate((cv2.split(cv2.resize(crop_img, size))), axis=1)
        flat_thumb = thumb_img.flatten()

        key = cls + str(lbl)
        if key in bbox_dict:
            bbox_dict[key] = np.vstack((bbox_dict[key], flat_thumb))
        else:
            bbox_dict[key] = thumb_img

    return bbox_dict

# TODO: Hierarchical K-Means clustering but how many K to be taken not mentioned.
def genClusterTress(bbox_dict, 5, )
#

