import keras
import cv2
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

#import NoiselessJointPPGN as PPGN

#Test on MNIST for now

batch_size = 32
num_classes = 10
epochs = 15
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
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

g_gen = Sequential()
g_gen.add(Dense(1600, activation='relu', input_shape=(64,)))
g_gen.add(Reshape((5,5,64)))
g_gen.add(Conv2DTranspose(512, (5,5),   activation='relu', padding='valid'))
g_gen.add(Conv2DTranspose(256, (5,5),   activation='relu', padding='valid'))
g_gen.add(Conv2DTranspose(256, (7,7),   activation='relu', padding='valid'))
g_gen.add(Conv2DTranspose(1,   (10,10), activation='linear', padding='valid'))

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

ppgn = PPGN.NoiselessJointPPGN(model, 6, 7, 8, verbose=2,
                               gan_generator='Default', gan_discriminator='Default')
#                               gan_generator=g_gen, gan_discriminator=g_disc)
ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[1, 1e-1, 2])
ppgn.fit_classifier(x_train, y_train, validation_data=[x_test, y_test], epochs=15)

src, gen = ppgn.fit_gan(x_train, epochs=5000)

#Plot the losses
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
    cv2.imwrite('img/gan{}.bmp'.format(i), img)

#h2_base = ppgn.enc2.predict(ppgn.enc1.predict(x_test[0:1]))
h2_base=None
for i in range(10):
    samples, h2 = ppgn.sample(i, nbSamples=100,
                              h2_start=h2_base,
                              epsilons=(1e-2, 1, 1e-15),
                              lr=1e24, lr_end=1e24)
    h2_base = None#h2[-1]
    img = (np.concatenate((samples), axis=0)+1)*255/2
    img[img < 0  ] = 0
    img[img > 255] = 255
    cv2.imwrite('img/samples{}.bmp'.format(i), img)
