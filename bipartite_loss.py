#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.image as mpimpg 
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow
import networkx as nx
from networkx.algorithms import bipartite


# In[2]:


# inst_gt=mpimpg.imread('./sample images/inst1.png')
# seg_gt=mpimpg.imread('./sample images/seg1.png')
# inst_pred=mpimpg.imread('./sample images/inst1.png')
# seg_pred=mpimpg.imread('./sample images/seg1.png')


# In[3]:


def get_mask(Y):
    unique_colors=set( tuple(v) for m2d in Y for v in m2d )
    border_pixels=max(unique_colors)
    #print(border_pixels)
    unique_colors.remove((0.0, 0.0, 0.0))
    unique_colors.remove(max(unique_colors))
    unique_colors=list(unique_colors)
    #print(unique_colors)
    masks=np.zeros((Y.shape[0],Y.shape[1],len(unique_colors)),int)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if tuple(Y[i,j])==(0.0,0.0,0.0) or tuple(Y[i,j])== border_pixels:
            #print(i,j)
                pass
            else:
                idx=unique_colors.index(tuple(Y[i,j]))
                #print(i,j,idx)
                masks[i,j,idx]=1
    return masks


# In[4]:


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


# In[5]:


def get_color_map_dict():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    cmap = color_map()
    cmap=list(cmap[0:22])
    cmap=tuple(map(tuple, cmap))
    cmap_dict=dict(zip(cmap,labels))
    cmap_dict[(128,128,64)] = cmap_dict[(128, 64, 128)]
    del cmap_dict[(128, 64, 128)]
    return cmap_dict


# In[6]:


def get_labels_and_masks(inst_gt,seg_gt):
    cmap_dict=get_color_map_dict()
    gt_mask=get_mask(inst_gt)
    labels=list()
    for i in range(gt_mask.shape[2]):
        l=np.where(gt_mask[:,:,i]==1)
        labels.append(cmap_dict[tuple(seg_gt[l[0][0],l[1][0]]*255)])
    return labels,gt_mask


# In[7]:


def iou_score(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# In[11]:


def Maximum_Bipartite_graph(labels_gt,mask_gt,labels_pred,mask_pred):
    gt_set=np.arange((len(labels_gt)))
    pred_set=len(gt_set)+np.arange((len(labels_pred)))
    B = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(gt_set, bipartite=0)
    B.add_nodes_from(pred_set, bipartite=1)
    for i in range(len(gt_set)):
        for j in range(len(pred_set)):
            k=j+len(gt_set)
    #         print(i,k)
            if labels_gt[i]==labels_pred[j]:
                iou=iou_score(mask_gt[:,:,i],mask_pred[:,:,j])
            else:
                iou=0
#             print(i,k,iou)
            B.add_edges_from([(i, k, {'iou': iou})])
    max_bipart_graph = nx.bipartite.maximum_matching(B)
    return max_bipart_graph


# In[21]:


def CrossEntropy(yHat, y):
    if y == 1:
          return -np.log(yHat)
    else:
          return -np.log(1 - yHat)


# In[24]:


# CrossEntropy(1,0)


# In[9]:


# labels_gt,mask_gt=get_labels_and_masks(inst_gt,seg_gt)
# labels_pred,mask_pred=get_labels_and_masks(inst_pred,seg_pred)


# In[12]:


# mbp=Maximum_Bipartite_graph(labels_gt,mask_gt,labels_pred,mask_pred)


# In[15]:





# In[ ]:





# In[ ]:




