import numpy as np
import scipy
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Dropout, Dense, Flatten
import os.path as osp

data_root = '../CS446-project_data'
Y = np.load(osp.join(data_root, 'train_binary_Y.npy'))
train_feature = np.load(osp.join(data_root, 'train_X.npy'))
d_x = len(train_feature[0])
d_y = len(train_feature[0][0])
d_z = len(train_feature[0][0][0])
train_feature = train_feature.reshape((len(train_feature), d_x, d_y, d_z, 1))


# num_classes = 45

# Y = np.zeros((train_feature.shape[0],num_classes))
# label_dict2 = {}
# cou = 0
# for i in range(train_label.shape[0]):
#     if tuple(train_label[i]) not in label_dict2.keys():
#         temp = [0 for i in range(45)]
#         temp[cou] = 1
#         cou += 1
#         label_dict2[tuple(train_label[i])] = tuple(temp)
#         Y[i] = temp
#     else:
#         Y[i] = label_dict2[tuple(train_label[i])]



# inv_map = {v: k for k, v in label_dict2.items()}

model = Sequential()
model.add(Conv3D(32, kernel_size=(6, 6, 6), strides=(2, 2, 2), input_shape=(d_x, d_y, d_z, 1), padding='same', activation='relu', data_format='channels_last'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(20, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# Fully connected
model.add(Flatten())
model.add(Dense(8000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(19, activation='sigmoid'))
model.summary()



# model = Sequential()
# model.add(Conv3D(16, kernel_size=(8, 8, 8), strides=(2, 2, 2), input_shape=(d_x, d_y, d_z, 1), padding='same', activation='relu', data_format='channels_last'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(32, kernel_size=(6, 6, 6), strides=(1, 1, 1), padding='same', activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# # model.add(Conv3D(1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# # Fully connected
# model.add(Flatten())
# model.add(Dense(5000, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(45, activation='sigmoid'))
# model.summary()

# model = Sequential()
# model.add(Conv3D(1, kernel_size=(3, 3, 3), strides=(1, 1, 1), input_shape=(d_x, d_y, d_z, 1), padding='same', activation='relu', data_format='channels_last'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# # model.add(Conv3D(1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# # Fully connected
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(45, activation='relu'))
# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_feature, Y, validation_split=0.15, epochs=100, batch_size=256, shuffle=True)

x_test = np.load(osp.join(data_root, 'valid_test_X.npy'))
x_test = x_test.reshape((len(x_test), len(x_test[0]), len(x_test[0][0]), len(x_test[0][0][0]), 1))
output = model.predict(x_test)

# for i in range(len(output)):
# 	big = np.argmax(output[i])
# 	output[i] = np.zeros(len(output[i]))
# 	output[i,big] = 1

# gg = np.zeros((output.shape[0],19))
# for i in range(output.shape[0]):
#     gg[i] = list(inv_map[tuple(output[i])])


np.save('result_cnn_2.npy',output)




