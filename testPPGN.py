import keras
import cv2
import numpy as np
from keras.datasets import mnist

from sklearn.preprocessing import Normalizer

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D, Conv2DTranspose,  Reshape

#import NoiselessJointPPGN as PPGN

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


#Classifier
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


#Encoder 1 x-->Flatten
enc1_input = Input(input_shape)
enc1_output = enc1_input
start=1
stop=6
for i in np.arange(start, stop, 1):
    enc1_output = model.get_layer(index=i)(enc1_output)
enc1 = Model(inputs=enc1_input, outputs=enc1_output)
enc1.trainable=False


#Encoder 2 Flatten-->Dense 128
enc2_input = Input((enc1.output_shape[1],))
enc2_output = enc2_input
start=6
stop=8
for i in np.arange(start, stop, 1):
    enc2_output = model.get_layer(index=i)(enc2_output)
enc2 = Model(inputs=enc2_input, outputs=enc2_output)
enc2.trainable=False


#GAN-generator
g_gen_input = Input((enc2.output_shape[1],))
g_gen_output = g_gen_input
g_gen_output = Dense(9216, activation='relu')(g_gen_output)
g_gen_output = Reshape((12,12,64))(g_gen_output)
g_gen_output = Dropout(0.25)(g_gen_output)
g_gen_output = UpSampling2D((2,2))(g_gen_output)
g_gen_output = Conv2DTranspose(32, (3,3), activation='relu')(g_gen_output)
g_gen_output = Conv2DTranspose(1, (3,3), activation='sigmoid')(g_gen_output)
g_gen = Model(inputs=g_gen_input, outputs=g_gen_output)


#GAN-discriminator
g_disc_input = Input(input_shape)
g_disc_output = g_disc_input
g_disc_output = Conv2D(32, (3,3), activation='relu')(g_disc_output)
g_disc_output = Conv2D(64, (3,3), activation='relu')(g_disc_output)
g_disc_output = MaxPooling2D((2,2))(g_disc_output)
g_disc_output = Dropout(0.25)(g_disc_output)
g_disc_output = Flatten()(g_disc_output)
g_disc_output = Dense(128, activation='relu')(g_disc_output)
g_disc_output = Dropout(0.5)(g_disc_output)
g_disc_output = Dense(2, activation='softmax')(g_disc_output)
g_disc = Model(inputs=g_disc_input, outputs=g_disc_output)
g_disc.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta(),
               metrics=['accuracy'])
g_disc.trainable=False


#Now the big Ugly, mega NET
full_input = Input(input_shape)
#1st output
full_output_img = full_input
full_output_img = enc1(full_input)
full_output_img = enc2(full_output_img)
full_output_img = g_gen(full_output_img)
#2nd output
full_output_disc = full_output_img
full_output_disc = g_disc(full_output_disc)
#3rd output
full_output_enc = full_output_img
full_output_enc = enc1(full_output_enc)
#Create the model and compile
full = Model(inputs=full_input, outputs=[full_output_img, full_output_disc, full_output_enc])
full.compile(optimizer=keras.optimizers.Adadelta(),
             metrics=None,
             loss=[keras.losses.mean_squared_error, 
                   keras.losses.categorical_crossentropy,
                   keras.losses.mean_squared_error],
             loss_weights=[1., 1., 1.])




model.summary()
enc1.summary()
enc2.summary()
g_gen.summary()
g_disc.summary()
full.summary()


#Train Step
#Classifier
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#Generate encoded version (enc1) of the dataset
h_train = enc1.predict(x_train)
h_test = enc1.predict(x_test)

#GAN
report_freq = 500
half_batch = 32
g_epochs=30000
disc_loss = []
gen_loss = []
for e in range(g_epochs):
    #Train the discriminator
    idX = np.random.randint(0, x_train.shape[0], half_batch)
    
    valid = x_train[idX]
    fake = g_gen.predict(enc2.predict(h_train[idX]))
    
    v_ret = g_disc.train_on_batch(valid, keras.utils.to_categorical(np.ones ((half_batch)), 2))
    f_ret = g_disc.test_on_batch(fake,   keras.utils.to_categorical(np.zeros((half_batch)), 2))
    disc_loss.append(v_ret[0]*0.5 + f_ret[0]*0.5)
    
    #Train the GAN on a full batch
    idX = np.random.randint(0, x_train.shape[0], 2*half_batch)
    full_loss = full.train_on_batch(x_train[idX], [x_train[idX], 
                                                   keras.utils.to_categorical(np.ones((2*half_batch)), 2),
                                                   h_train[idX]])
    gen_loss.append(full_loss)
    
    if e % report_freq == 0:
        print(':::: Epoch #{} report ::::'.format(e))
        print('GAN losses --- disc: {:.2f} // gen:{:.2f}'.format(disc_loss[-1], gen_loss[-1][2]))
        print('Reconstruction losses --- img: {:.2f} // h1: {:.2f}'.format(gen_loss[-1][1], gen_loss[-1][3]))
        
        saveImg = np.concatenate((valid, fake), axis=2)
        saveImg = np.vstack(saveImg)
        saveImg = saveImg*255
        saveImg[saveImg<0] = 0
        saveImg[saveImg>255] = 255
        cv2.imwrite('epoch_{}.bmp'.format(e), saveImg)
        
    
    






























