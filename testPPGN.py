import keras
from keras.datasets import mnist

from sklearn.preprocessing import Normalizer

import NoiselessJointPPGN as PPGN

#Test on MNIST for now

batch_size = 128
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28

#Get the data and reshape/convert/normalize
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Take the min/max over each channel and normalize
#Here it just means we divide by 256
mini = x_train.min(axis=(0,1,2), keepdims=True)
maxi = x_train.max(axis=(0,1,2), keepdims=True)
x_train = (x_train - mini) / (maxi - mini)
x_test =  (x_test  - mini) / (maxi - mini)

#categorical y
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)












ppgn = PPGN.NoiselessJointPPGN()




