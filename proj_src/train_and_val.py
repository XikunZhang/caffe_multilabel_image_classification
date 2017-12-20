# coding: utf-8

# # Multilabel classification on PASCAL using python data-layers

# In this tutorial we will do multilabel classification on PASCAL VOC 2012.
#
# Multilabel classification is a generalization of multiclass classification, where each instance (image) can belong to many classes. For example, an image may both belong to a "beach" category and a "vacation pictures" category. In multiclass classification, on the other hand, each image belongs to a single class.
#
# Caffe supports multilabel classification through the SigmoidCrossEntropyLoss layer, and we will load data using a Python data layer. Data could also be provided through HDF5 or LMDB data layers, but the python data layer provides endless flexibility, so that's what we will use.

# ### 1. Preliminaries
#
# * First, make sure you compile caffe using
# WITH_PYTHON_LAYER := 1
#
# * Second, download PASCAL VOC 2012. It's available here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
#
# * Third, import modules:

# In[1]:


import sys
import os

import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

import create_nets_and_solver as cnas

caffe_root = '~/caffe'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root + 'python')
import caffe  # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P  # Shortcuts to define the net prototxt.

sys.path.append("pycaffe/layers")  # the datalayers we will use are in this directory.
sys.path.append("pycaffe")  # the tools file is in this folder

import tools  # this contains some tools that we need

# * Fourth, set data directories and initialize caffe

# In[8]:
caffe.set_device(0)
caffe.set_mode_gpu()

cur_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.dirname(cur_dir)
# set data root directory, e.g:
data_root = osp.join(parent_dir, 'CS446-project_data')
workdir = osp.join(cur_dir, 'models')


class CNN(object):
    def __init__(self, train_idx, valid_idx):
        cnas.write_nets(train_idx, valid_idx)

        cnas.write_solver(
            display="1",
            test_iter="100",
            test_interval="25000",
            base_lr="0.001",
            momentum="0.9",
            momentum2="0.999",
            lr_policy="\"step\"",
            gamma="0.1",
            stepsize="10000",
            max_iter="100000",
            weight_decay="0.0005",
            snapshot="2500",
            type="\"Adam\""
        )

        self.solver = caffe.AdamSolver(osp.join(workdir, 'solver.prototxt'))
        # solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
        self.solver.test_nets[0].share_with(self.solver.net)
        # for itt in range(6):
        #     solver.step(100)
        #     print('itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50)))
        print([(k, v.data.shape) for k, v in self.solver.net.blobs.items()])
        print([(k, v[0].data.shape) for k, v in self.solver.net.params.items()])

    # def train(self, niter, test_iter, test_interval=None):
    #     self.solver.step(niter)
    #     return check_accuracy(self.solver.test_nets[0], test_iter)
    def train(self, niter, test_interval, test_iter, idx):
        # losses will also be stored in the log
        train_loss = np.zeros(niter)
        train_acc = np.zeros(int(np.ceil(niter / test_interval)))
        test_acc = np.zeros(int(np.ceil(niter / test_interval)))

        train_correct = 0
        train_batch_size = len(self.solver.net.blobs['label'].data)
        # the main solver loop
        for it in range(niter):
            self.solver.step(1)  # SGD by Caffe
            train_loss[it] = self.solver.net.blobs['loss'].data
            gts = self.solver.net.blobs['label'].data
            ests = self.solver.net.blobs['score'].data > 0
            for gt, est in zip(gts, ests):
                train_correct += np.array_equal(gt, est)

            if it % test_interval == 0:
                print('Iteration', it, 'testing...')
                test_acc[it // test_interval] = check_accuracy(self.solver.test_nets[0], test_iter)
                train_acc[it // test_interval] = train_correct / (train_batch_size * test_interval)
                train_correct = 0
        # _, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        # # ax1.plot(np.arange(niter), train_loss)
        # ax1.plot(test_interval * np.arange(len(train_acc)), train_acc, 'b')
        # ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
        # ax1.set_xlabel('iteration')
        # # ax1.set_ylabel('train loss')
        # ax1.set_ylabel('train accuracy')
        # ax2.set_ylabel('test accuracy')
        # ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))

        plt.clf()
        plt.plot(test_interval * np.arange(len(train_acc)), train_acc, 'b', label='train')
        plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r', label='validate')
        plt.xlabel('iteration')
        # ax1.set_ylabel('train loss')
        plt.ylabel('accuracy')
        plt.legend()
        plt.title('Test Accuracy: {:.2f}'.format(np.mean(test_acc[-5:])))
        plt.savefig('./models/learning_curve_{}.png'.format(idx))


def hamming_similarity(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))


def check_accuracy(net, num_batches, metric='accuracy_score'):
    batch_size = len(net.blobs['label'].data)
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            # 'gt' and 'est' are two 1-d arrays
            if metric == 'accuracy_score':
                acc += np.array_equal(gt, est)
            else:
                acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


def tenfold_cv():
    num_all = 4602
    valid_acc = []
    kf = KFold(n_splits=)
    for idx, tpl in enumerate(kf.split(range(num_all))):
        train, valid = tpl
        cnn = CNN(train, valid)
        niter = 3000
        acc = cnn.train(
            niter=niter,
            # test_iter = len(valid),
            test_interval=100,
            test_iter=100,
            idx=idx
        )
    #     valid_acc.append(acc)
    # print('valid_acc: ', valid_acc)
    # return np.mean(valid_acc)


if __name__ == '__main__':
    # print(tenfold_cv())
    tenfold_cv()

