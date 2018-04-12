import keras
import cv2
import os, h5py
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from matplotlib import pylab as plt

import NoiselessJointPPGN as PPGN

from sklearn.preprocessing import Normalizer

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D, Conv2DTranspose,  Reshape

## Definbe custom GAN training procedure based on http://www.nada.kth.se/~ann/exjobb/hesam_pakdaman.pdf
#Do 5 disc iterations for one gan iteration. Except for the 500 first epoch and every 500 subsequent epochs
#where disc is trained 100 times
#Based on implementation found in https://github.com/hesampakdaman/ppgn-disc/blob/master/src/vanilla.py
def customGANTrain(x_train, h1_train, batch_size, disc_model, gan_model, epochID):
    disc_train = 100 if epochID < 25 or epochID % 500 else 5

    #train disc
    idX = np.random.randint(0, x_train.shape[0], disc_train)

    valid = x_train[idX]
    fake  = gan_model.predict(x_train[idX])[0]
    x_disc = np.concatenate((valid, fake), axis=0)
    y_disc = np.concatenate((np.ones((disc_train)), np.zeros((disc_train))))

    disc_loss = disc_model.train_on_batch(x_disc, y_disc)

    #train gen
    x_gan = x_train[idX][-1:]
    y_gan = np.ones((1))
    h1_gan = h1_train[idX][-1:]

    gan_loss = gan_model.train_on_batch(x_gan, [x_gan, y_gan, h1_gan])

    return (disc_loss, gan_loss)


#Test on MNIST for now
batch_size = 32
num_classes = 14
epochs = 15
# input image dimensions
img_rows, img_cols = 28, 28
data_path = '/home/romain/Projects/cda_bn2018/data/h5py/'
fname = '/gray_%ix%i.hdf5' %(img_rows, img_cols)
#Get the data and reshape/convert/normalize
data = h5py.File(data_path + fname, "r")
n_train = len(data['x_train'])
idx = np.arange(n_train)
np.random.shuffle(idx)
x_train = np.array(data['x_train'])[idx]
y_train = np.array(data['y_train'])[idx]
n_test = len(data['x_test'])
idx = np.arange(n_test)
np.random.shuffle(idx)
x_test = np.array(data['x_test'])[idx]
y_test = np.array(data['y_test'])[idx]
n_train, n_test = y_train.shape[0], y_test.shape[0]
print('train images: %i' %n_train)
print('test images: %i' %n_test)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Take the min/max over each channel and normalize
#Here it just means we divide by 256
x_train = ((x_train/255) - 0.5)*2
x_test =  ((x_test/255) - 0.5)*2

#categorical y
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Classifier
model = Sequential()
model.add(Conv2D(64, (7,7), activation='relu', input_shape=input_shape, padding='valid'))
model.add(Conv2D(128, (7,7), activation='relu', padding='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (7,7), activation='relu', padding='valid'))
model.add(MaxPooling2D((2,2)))
#model.add(Conv2D(256, (7,7), activation='relu', padding='valid'))
#model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.trainable=True

g_gen = Sequential()
g_gen.add(Dense(1600, activation='relu', input_shape=(64,)))
g_gen.add(Reshape((5,5,64)))
g_gen.add(Conv2DTranspose(512, (5,5),   activation='relu', padding='valid'))
g_gen.add(Conv2DTranspose(256, (5,5),   activation='relu', padding='valid'))
g_gen.add(Conv2DTranspose(256, (7,7),   activation='relu', padding='valid'))
g_gen.add(Conv2DTranspose(1,   (10,10), activation='linear', padding='valid'))
g_gen.trainable=True
# g_gen.add(Conv2DTranspose(512, (5,5), activation='relu', padding='valid'))
# g_gen.add(Conv2DTranspose(256, (5,5), activation='relu', padding='valid'))
# g_gen.add(UpSampling2D((2, 2)))
# g_gen.add(Conv2DTranspose(256, (5,5), activation='relu', padding='valid'))
# g_gen.add(UpSampling2D((2, 2)))
# g_gen.add(Conv2DTranspose(1, (5,5), activation='linear', padding='valid'))

g_disc = Sequential()
g_disc.add(Conv2D(256, (3,3), activation='relu', input_shape=input_shape, padding='valid'))
g_disc.add(Conv2D(256, (3,3), activation='relu', padding='valid'))
g_disc.add(MaxPooling2D((2,2)))
g_disc.add(Conv2D(256, (3,3), activation='relu', padding='valid'))
g_disc.add(MaxPooling2D((2,2)))
g_disc.add(Conv2D(512, (3,3), activation='relu', padding='valid'))
g_disc.add(MaxPooling2D((2,2)))
g_disc.add(Flatten())
g_disc.add(Dense(1, activation='linear'))
g_disc.trainable=True

#Load weights and skip fit if possible
skipFitClf=False
skipFitGAN=False
if skipFitClf and 'clf_feuilles.h5' in os.listdir('weights/'):
    model.load_weights('weights/clf_feuilles.h5')
    skipFitClf=True
    print('Loaded CLF weights from existing file, will skip training')
if skipFitGAN and 'g_gen_feuilles.h5' in os.listdir('weights/') and 'g_disc_feuilles.h5' in os.listdir('weights/'):
    g_gen.load_weights('weights/g_gen_feuilles.h5')
    g_disc.load_weights('weights/g_disc_feuilles.h5')
    skipFitGAN=True
    print('Loaded GAN weights from existing file, will skip training')


ppgn = PPGN.NoiselessJointPPGN(model, 6, 7, 8, verbose=2,
#                               gan_generator='Default', gan_discriminator='Default')
                               gan_generator=g_gen, gan_discriminator=g_disc)

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[1, 2, 1e-1])

if not skipFitClf:
    print('Fitting classifier')
    ppgn.fit_classifier(x_train, y_train, validation_data=[x_test, y_test], epochs=15)
    ppgn.classifier.save_weights('weights/clf_feuilles.h5')

if not skipFitGAN:
    print('Fitting GAN')
    src, gen = ppgn.fit_gan(x_train, epochs=2500, report_freq=100)#, train_procedure=customGANTrain)
    #src, gen = ppgn.fit_gan(x_train, epochs=2500, report_freq=100, train_procedure=customGANTrain)
    ppgn.g_gen.save_weights('weights/g_gen_feuilles.h5')
    ppgn.g_disc.save_weights('weights/g_disc_feuilles.h5')

    #Plot some GAN metrics computed during fit
    plt.ion()
    plt.figure()
    plt.plot(np.array(ppgn.g_disc_loss))
    plt.plot(np.array(ppgn.gan_loss)[:, 2])
    plt.legend(['disc_loss', 'gen_loss'])
    plt.figure()
    plt.plot(np.array(ppgn.gan_loss))
    plt.legend(['total loss', 'img loss', 'gan loss', 'h loss'])

    for i in range(len(src)):
        src[i] = np.concatenate((src[i]), axis=0)
        gen[i] = np.concatenate((gen[i]), axis=0)
        img = (np.concatenate((src[i], gen[i]), axis=1)+1)*255/2
        img[img < 0  ] = 0
        img[img > 255] = 255
        cv2.imwrite('img/feuilles_gan{}.bmp'.format(i), img)

h2_base = ppgn.enc2.predict(ppgn.enc1.predict(x_test[0:1]))
h2_base=None
for i in range(num_classes):
    samples, h2 = ppgn.sample(i, nbSamples=100,
                              h2_start=h2_base,
                              epsilons=(1e-2, 1, 1e-15),
                              lr=.5, lr_end=.5)
    h2_base = None#h2[-1]
    img = (np.concatenate((samples), axis=0)+1)*255/2
    img[img < 0  ] = 0
    img[img > 255] = 255
    img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
    fname = 'img/feuilles_{}x{}samples{}.bmp'.format(input_shape[0], input_shape[1], i)
    cv2.imwrite(fname, img_grid)
