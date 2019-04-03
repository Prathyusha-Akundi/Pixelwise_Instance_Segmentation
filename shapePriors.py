import numpy as np
from numpy import linalg as la
import cv2
import scipy.io as spio
from os import listdir


###################### FUNCTIONS


# Get all the bounding boxes and related information.
# Input:
#   seg_dict          Dictionary obtained from the segmentation sub-network.
# Outputs:
#   bbox_img_name     Name of the images to which bounding boxes belong. Vector of size = tot_num_of_bbox.
#   bbox_cls          Class of the bounding boxes. Vector of size = tot_num_of_bbox.
#   bbox_coord        Coordinates of the bounding boxes (x1,y1,x2,y2) = Matrix of size = (tot_num_of_bbox, 4).
#   bbox_score        Confidence score of the bounding boxes. Vector of size = tot_num_of_bbox.
def bbox_data(seg_dict):
    bbox_img_name = []
    bbox_coord = np.empty((0, 4), int)
    bbox_score = []
    bbox_cls = []

    for img_name in seg_dict:
        img_seg_data = seg_dict[img_name]
        seg_keys = list(seg_dict[img_name])

        num_seg = img_seg_data[seg_keys[0]]
        bbox_img_name = list(np.hstack((bbox_img_name, [img_name for i in range(num_seg)])))
        bbox_coord = np.vstack((bbox_coord, img_seg_data[seg_keys[1]]))
        bbox_score = list(np.hstack((bbox_score, img_seg_data[seg_keys[2]])))
        bbox_cls = list(np.hstack((bbox_cls, img_seg_data[seg_keys[3]])))

    return bbox_img_name, bbox_cls, bbox_coord, bbox_score


# Generates a dictionary of shape templates. The mat file should be kept in the same folder as the code.
# Input:
#   mat_filename    Name of the matlab file containing shape templates.
# Output:
#   shape_temps     Dictionary with Key = aspect ration(ar) and Value = Templates belonging to the ar.
def get_shape_templates(mat_filename):
    mat_file = spio.loadmat(mat_filename)
    models = mat_file['models']

    shape_temps = {}
    for idx in range(0, models.shape[1]):
        model = models[0, idx]
        shape_temps[model['ar'][0][0]] = model['shapes']

    return shape_temps


# Creates dictionary of images present in a folder
# Input:
#   path    Folder containing images
# Output:
#   img_dict    Dictionary with Key = Name of the image and Value = Normalized image in RGB format
def create_img_dict(path):
    img_dict = {}

    img_list = listdir(path)
    for img_name in img_list:
        img_dict[path + '/' +img_name] = cv2.cvtColor(cv2.imread(path + '/' + img_name), cv2.COLOR_BGR2RGB) / 255

    return img_dict


# Generates shape terms for all bounding boxes.
# Inputs:
#   coords      Matrix containing coordinates (x1,y1,x2,y2) for all the bounding boxes. Size = (tot_num_of_bbox, 4).
#   img_names   Name of the images bounding boxes correspond to. Size = tot_num_of_bbox.
#   img_dict    Dictionary with Key = Name of the image and Value = Normalized image in RGB format.
#   templates   Dictionary with Key = aspect ration(ar) and Value = Templates belonging to the ar.
# Output:
#   shape_terms Shape term for each bounding box. Vector of size = tot_num_of_bbox.
def gen_shape_terms(coords, img_names, img_dict, templates):
    shape_terms = []
    for i in range(0, 2):
        img = img_dict[img_names[i]]
        r, c = img.shape[0:2]
        coord = coords[0]
        bb = img[int(round(coord[0] * c)): int(round(coord[2] * c)), int(round(coord[1] * r)): int(round(coord[3] * r))]
        shape_terms.append(gen_shape_term(bb, templates))

    return shape_terms


# Provides shape potential for a given bounding box.
# Input:
#   bbox        Bounding box (2D matrix).
#   shape_temps Dictionary of shape templates.
# Output:
#   shape_cost  Shape term for bounding box.
def gen_shape_term(bbox, shape_temps):
    rows, cols = bbox.shape[0], bbox.shape[1]

    match_temp = []
    max_corr = []
    for idx in shape_temps:
        temps = shape_temps[idx]
        temps = cv2.resize(temps, (cols, rows), interpolation=cv2.INTER_LINEAR)

        prod = np.add(np.einsum('ij,ijk->ijk',bbox[:, :, 0], temps), np.einsum('ij,ijk->ijk',bbox[:, :, 1], temps))
        prod = np.add(prod, np.einsum('ij,ijk->ijk',bbox[:, :, 2], temps))
        num = np.sum(prod, axis=(0, 1))
        deno = la.norm(bbox) * la.norm(temps, axis=(0, 1))
        corr = num / deno
        match_temp.append(np.argmax(corr))
        max_corr.append(np.max(corr))

    arg_max_corr = np.argmax(max_corr)

    temps = shape_temps[list(shape_temps.keys())[arg_max_corr]]
    matched_temp = cv2.resize(temps[:, :, match_temp[arg_max_corr]], (cols, rows), interpolation=cv2.INTER_LINEAR)
    shape_cost = np.add(np.multiply(bbox[:, :, 0], matched_temp), np.multiply(bbox[:, :, 1], matched_temp))
    shape_cost = np.add(shape_cost,  np.multiply(bbox[:, :, 2], matched_temp))

    return shape_cost


