# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class MultilabelDataLayerSync(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        # check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 26, 31, 23)
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(self.batch_size, 19)

        # print_info("MultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            # "im" is a 4-D numpy array
            # "multilabel" is a 2-D numpy array
            im, multilabel = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.data_root = params['data_root']
        
        all_x = np.load(osp.join(self.data_root, 'train_X.npy'))
        all_y = np.load(osp.join(self.data_root, 'train_binary_Y.npy'))
        self.mean_x = np.mean(all_x, axis=0)
        np.save(osp.join(self.data_root, 'mean_X.npy'), self.mean_x)

        # num_all = len(all_x)
        # num_valid = num_all // 6
        # idx_list = np.arange(num_all)
        # shuffle(idx_list)
        # valid_idx_list = idx_list[:num_valid]
        # train_idx_list = idx_list[num_valid:]
        # train_x = all_x[train_idx_list]
        # train_y = all_y[train_idx_list]
        # valid_x = all_x[valid_idx_list]
        # valid_y = all_y[valid_idx_list]
        
        # if params['split'] == 'train':
        #     # self.x = train_x
        #     # self.y = train_y
        #     self.x = all_x
        #     self.y = all_y
        # else:
        #     self.x = valid_x
        #     self.y = valid_y
        self.x = all_x[params['idx']]
        self.y = all_y[params['idx']]
        self._cur = -1  # current image
        # this class does some simple data-manipulations
        # self.transformer = SimpleTransformer()

        print ("BatchLoader initialized with {} images".format(
            len(self.x)))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        self._cur += 1
        # Did we finish an epoch?
        if self._cur == len(self.x):
        # if self._cur == 20:
            self._cur = 0
            # shuffle(self.indexlist)

        # Load an image
        # index = self.indexlist[self._cur]  # Get the image index
        # image_file_name = index + '.jpg'
        # im = np.asarray(Image.open(
        #     osp.join(self.pascal_root, 'JPEGImages', image_file_name)))
        # im = scipy.misc.imresize(im, self.im_shape)  # resize
# 
        # # do a simple horizontal flip as data augmentation
        # flip = np.random.choice(2)*2-1
        # im = im[:, ::flip, :]
# 
        # # Load and prepare ground truth
        # multilabel = np.zeros(20).astype(np.float32)
        # anns = load_pascal_annotation(index, self.pascal_root)
        # for label in anns['gt_classes']:
        #     # in the multilabel problem we don't care how MANY instances
        #     # there are of each class. Only if they are present.
        #     # The "-1" is b/c we are not interested in the background
        #     # class.
        #     multilabel[label - 1] = 1

        return self.x[self._cur] - self.mean_x, self.y[self._cur]


# def load_pascal_annotation(index, pascal_root):
#     """
#     This code is borrowed from Ross Girshick's FAST-RCNN code
#     (https://github.com/rbgirshick/fast-rcnn).
#     It parses the PASCAL .xml metadata files.
#     See publication for further details: (http://arxiv.org/abs/1504.08083).
# 
#     Thanks Ross!
# 
#     """
#     classes = ('__background__',  # always index 0
#                'aeroplane', 'bicycle', 'bird', 'boat',
#                'bottle', 'bus', 'car', 'cat', 'chair',
#                          'cow', 'diningtable', 'dog', 'horse',
#                          'motorbike', 'person', 'pottedplant',
#                          'sheep', 'sofa', 'train', 'tvmonitor')
#     class_to_ind = dict(zip(classes, range(21)))
# 
#     filename = osp.join(pascal_root, 'Annotations', index + '.xml')
#     # print 'Loading: {}'.format(filename)
# 
#     def get_data_from_tag(node, tag):
#         return node.getElementsByTagName(tag)[0].childNodes[0].data
# 
#     with open(filename) as f:
#         data = minidom.parseString(f.read())
# 
#     objs = data.getElementsByTagName('object')
#     num_objs = len(objs)
# 
#     boxes = np.zeros((num_objs, 4), dtype=np.uint16)
#     gt_classes = np.zeros((num_objs), dtype=np.int32)
#     overlaps = np.zeros((num_objs, 21), dtype=np.float32)
# 
#     # Load object bounding boxes into a data frame.
#     for ix, obj in enumerate(objs):
#         # Make pixel indexes 0-based
#         x1 = float(get_data_from_tag(obj, 'xmin')) - 1
#         y1 = float(get_data_from_tag(obj, 'ymin')) - 1
#         x2 = float(get_data_from_tag(obj, 'xmax')) - 1
#         y2 = float(get_data_from_tag(obj, 'ymax')) - 1
#         cls = class_to_ind[
#             str(get_data_from_tag(obj, "name")).lower().strip()]
#         boxes[ix, :] = [x1, y1, x2, y2]
#         gt_classes[ix] = cls
#         overlaps[ix, cls] = 1.0
# 
#     overlaps = scipy.sparse.csr_matrix(overlaps)
# 
#     return {'boxes': boxes,
#             'gt_classes': gt_classes,
#             'gt_overlaps': overlaps,
#             'flipped': False,
#             'index': index}


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'data_root']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print ("{} initialized for split: {}, with bs: {}.".format(
        name,
        params['split'],
        params['batch_size']))
