
import os.path as osp
import caffe
import numpy as np


caffe.set_mode_gpu()

snapshot_iter = 10000

cur_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.dirname(cur_dir)
# set data root directory, e.g:
data_root = osp.join(parent_dir, 'CS446-project_data')
# Read model architecture and trained model's weights
net = caffe.Net('./models/deploynet.prototxt',
                './snapshot_iter_{}.caffemodel'.format(snapshot_iter),
                caffe.TEST)


'''
Making predicitions
'''
test_x = np.load(osp.join(data_root, 'valid_test_X.npy'))
mean_x = np.load(osp.join(data_root, 'mean_X.npy'))
print('test_x.shape: ', test_x.shape)
print('mean_x.shape: ', mean_x.shape)

# Making predictions
predictions = np.empty((len(test_x), 19))
for idx, img in enumerate(test_x):
    net.blobs['data'].data[...] = img - mean_x
    net.forward()
    predictions[idx] = net.blobs['score'].data > 0

'''
Making submission file
'''
print('predictions.shape: ', predictions.shape)
np.save('./submissions/submission_1.npy', predictions)
