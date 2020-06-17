import os
import sys
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from layers.graph import Graph

import time


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self, data_path, img_path, graph_args={}, train_val_test='train'):
        '''
        train_val_test: (train, val, test)
        '''
        self.data_path = data_path
        self.img_path = img_path
        self.load_data()
        self.load_img()

        total_num = len(self.all_feature)
        # equally choose validation set
        train_id_list = list(np.arange(0, total_num).astype(int))

        # # last 20% data as validation set
        self.train_val_test = train_val_test
        self.all_feature = self.all_feature[train_id_list]
        self.all_adjacency = self.all_adjacency[train_id_list]
        self.all_mean_xy = self.all_mean_xy[train_id_list]
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

        self.graph = Graph(**graph_args) #num_node = 120,max_hop = 1

    def load_data(self):
        with open(self.data_path, 'rb') as reader:
            # Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
            [self.all_feature, self.all_adjacency, self.all_mean_xy]= pickle.load(reader)
            print("in loader data",self.all_adjacency.shape)

    def load_img(self):
        with open(self.img_path, 'r') as reader:
            self.all_img = [x.strip().split(' ') for x in reader.readlines()]

    def __len__(self):
        return len(self.all_feature)

    def __getitem__(self, idx):
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        now_feature = self.all_feature[idx].copy() # (C, T, V) = (11, 12, 120)
        now_mean_xy = self.all_mean_xy[idx].copy() # (2,) = (x, y) 

        if self.train_val_test.lower() == 'train' and np.random.random()>0.5:
            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            angle_mat = np.array(
                [[cos_angle, -sin_angle],
                [sin_angle, cos_angle]])

            xy = now_feature[3:5, :, :]
            num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data

            # angle_mat: (2, 2), xy: (2, 12, 120)
            out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
            now_mean_xy = np.matmul(angle_mat, now_mean_xy)
            xy[:,:,:num_xy] = out_xy[:,:,:num_xy]

            now_feature[3:5, :, :] = xy

        now_adjacency = self.graph.get_adjacency(self.all_adjacency[idx]) #(120,120)
        now_A = self.graph.normalize_adjacency(now_adjacency)#(3,120,120)
        
        ## Image Loader 
        ## Changes to be made to the images and channels
        now_img = plt.imread(self.all_img[idx][0])[:,:,:3] 
        now_img = (now_img - self.mean_rgb)/self.std_rgb  

        now_img = np.moveaxis(now_img, -1, 0)
        
        return now_feature, now_A, now_mean_xy, now_img

