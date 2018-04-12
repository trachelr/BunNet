import os, sys, h5py
import numpy as np

from skimage import io
from skimage.color import rgb2grey
from skimage.transform import resize

data_path = '/home/romain/Projects/cda_bn2018/data/'
base_path = data_path + 'BASE_IMAGE/Augmented_alpha1500/'
res_path = data_path + 'h5py/'

out_shape = [64, 64]
test_size = 600

class_names = os.listdir(base_path)
n_samples, n_classes = 0, len(class_names)
for c in class_names:
    n_samples += len(os.listdir(base_path + c))
    print(c, n_samples)

n_test_files = int(test_size * n_classes)
n_train_files = n_samples - n_test_files

fname = "gray_%ix%i.hdf5" %(out_shape[0], out_shape[1])
feuilles = h5py.File(res_path + fname, "w")

x_train = feuilles.create_dataset("x_train", (n_train_files, out_shape[0], out_shape[1]), dtype='i8')
y_train = feuilles.create_dataset("y_train", (n_train_files, ), dtype='i8')
x_test = feuilles.create_dataset("x_test", (n_test_files, out_shape[0], out_shape[1]), dtype='i8')
y_test = feuilles.create_dataset("y_test", (n_test_files, ), dtype='i8')

train_counter, test_counter = 0, 0
for dir_name in class_names:
    print(dir_name)
    fnames = os.listdir(os.path.join(base_path, dir_name))
    for fname in fnames[:-test_size]:
        im = io.imread(os.path.join(base_path, dir_name, fname))
        gray = rgb2grey(im) * 255
        res = resize(gray, out_shape)
        x_train[train_counter] = res
        y_train[train_counter] = int(dir_name.split('_')[0])
        train_counter += 1

    for fname in fnames[-test_size:]:
        im = io.imread(os.path.join(base_path, dir_name, fname))
        gray = rgb2grey(im) * 255
        res = resize(gray, out_shape)
        x_test[test_counter] = res
        y_test[test_counter] = int(dir_name.split('_')[0])
        test_counter += 1

feuilles.close()
