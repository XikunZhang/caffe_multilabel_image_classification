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
import matplotlib.pyplot as plt

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


cur_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.dirname(cur_dir)
# set data root directory, e.g:
data_root = osp.join(parent_dir, 'CS446-project_data')
workdir = osp.join(cur_dir, 'models')

# these are the PASCAL classes, we'll need them later.
# classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

# # make sure we have the caffenet weight downloaded.
# if not osp.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print("Downloading pre-trained CaffeNet model...")
#     os.system('~/caffe/scripts/download_model_binary.py ~/caffe/models/bvlc_reference_caffenet')




# ### 2. Define network prototxts
#
# * Let's start by defining the nets using caffe.NetSpec. Note how we used the SigmoidCrossEntropyLoss layer. This is the right loss for multilabel classification. Also note how the data layer is defined.

# In[9]:


# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, weight_filler=None, bias_filler=None, param=None):
    if weight_filler is None:
        weight_filler = {'type': 'gaussian', 'std': 0.01}
    if bias_filler is None:
        bias_filler = {'type': 'constant', 'value': 0}
    if param is None:
        param = [{'lr_mult': 1, 'decay_mult': 1}, {'lr_mult': 2, 'decay_mult': 0}]
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         weight_filler=weight_filler, bias_filler=bias_filler,
                         param=param)
    return conv, L.ReLU(conv, in_place=True)


# another helper function
def fc_relu(bottom, nout, weight_filler=None, bias_filler=None):
    if weight_filler is None:
        weight_filler = {'type': 'gaussian', 'std': 0.005}
    if bias_filler is None:
        bias_filler = {'type': 'constant', 'value': 1}
    fc = L.InnerProduct(bottom, num_output=nout, weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


# # main netspec wrapper
# def caffenet_multilabel(data_layer_params, datalayer):
#     # setup the python data layer
#     n = caffe.NetSpec()
#     n.data, n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer,
#                                ntop = 2, param_str=str(data_layer_params))

#     # the net itself
#     n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
#     n.pool1 = max_pool(n.relu1, 3, stride=2)
#     n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
#     n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
#     n.pool2 = max_pool(n.relu2, 3, stride=2)
#     n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
#     n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
#     n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
#     n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
#     n.pool5 = max_pool(n.relu5, 3, stride=2)
#     n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
#     n.drop6 = L.Dropout(n.relu6, in_place=True)
#     n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
#     n.drop7 = L.Dropout(n.relu7, in_place=True)
#     n.score = L.InnerProduct(n.drop7, num_output=20)
#     n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)

#     return str(n.to_proto())


# In[10]:


# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer):
    # setup the python data layer
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='multilabel_datalayers', layer=datalayer,
                               ntop=2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 1, 96, stride=1, weight_filler={'type': 'xavier'})
    n.pool1 = max_pool(n.relu1, 2, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 1, 128, group=2, weight_filler={'type': 'xavier'})
    n.pool2 = max_pool(n.relu2, 2, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    #     n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    #     n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    #     n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    #     n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.norm2, 300, weight_filler={'type': 'xavier'})
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 300, weight_filler={'type': 'xavier'})
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=19)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)

    return str(n.to_proto())


# ### 3. Write nets and solver files
#
# * Now we can create net and solver prototxts. For the solver, we use the CaffeSolver class from the "tools" module

# In[11]:


def write_nets():
    if not os.path.isdir(workdir):
        os.makedirs(workdir)


    # write train net.
    with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
        # provide parameters to the data layer as a python dictionary. Easy as pie!
        data_layer_params = dict(batch_size=64, split='train', data_root=data_root)
        f.write(caffenet_multilabel(data_layer_params, 'MultilabelDataLayerSync'))

    # write validation net.
    with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
        data_layer_params = dict(batch_size=64, split='valid', data_root=data_root)
        f.write(caffenet_multilabel(data_layer_params, 'MultilabelDataLayerSync'))

# * This net uses a python datalayer: 'PascalMultilabelDataLayerSync', which is defined in './pycaffe/layers/pascal_multilabel_datalayers.py'.
#
# * Take a look at the code. It's quite straight-forward, and gives you full control over data and labels.


def write_solver(**kwargs):
    solverprototxt = tools.CaffeSolver(trainnet_prototxt_path=osp.join(workdir, "trainnet.prototxt"),
                                       testnet_prototxt_path=osp.join(workdir, "valnet.prototxt"))
    for key, value in kwargs.items():
        solverprototxt.sp[key] = value
    solverprototxt.write(osp.join(workdir, 'solver.prototxt'))