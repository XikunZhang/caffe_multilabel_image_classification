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

import create_nets_and_solver as cnas

# get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (6, 6)

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

cnas.write_nets()

cnas.write_solver(
    display = "1",
    test_iter = "100",
    test_interval = "250",
    base_lr = "0.001",
    momentum = "0.9",
    momentum2 = "0.999",
    lr_policy = "\"step\"",
    gamma = "0.1",
    stepsize = "10000",
    max_iter = "100000",
    weight_decay = "0.0005",
    snapshot = "2500",
    type = "\"Adam\""
)


solver = caffe.AdamSolver(osp.join(workdir, 'solver.prototxt'))
# solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
solver.test_nets[0].share_with(solver.net)

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

# for itt in range(6):
#     solver.step(100)
#     print('itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50)))
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])

niter = 1000
test_interval = 25
test_iter = 50
# losses will also be stored in the log
train_loss = np.zeros(niter)
train_acc = np.zeros(int(np.ceil(niter / test_interval)))
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
# output = np.zeros((niter, 8, 10))

train_correct = 0
train_batch_size = len(solver.net.blobs['label'].data)
# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    gts = solver.net.blobs['label'].data
    ests = solver.net.blobs['score'].data > 0
    for gt, est in zip(gts, ests):
        train_correct += np.array_equal(gt, est)
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    # solver.test_nets[0].forward(start='conv1')
    # output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print('Iteration', it, 'testing...')
        # correct = 0
        # for test_it in range(100):
        #     solver.test_nets[0].forward()
        #     print(solver.test_nets[0].blobs['label'].data.shape)
        #     correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
        #                    == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = check_accuracy(solver.test_nets[0], test_iter)
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

plt.plot(test_interval * np.arange(len(train_acc)), train_acc, 'b', label='train')
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r', label='validate')
plt.xlabel('iteration')
# ax1.set_ylabel('train loss')
plt.ylabel('accuracy')
plt.legend()
plt.title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.savefig('./models/learning_curve.png')