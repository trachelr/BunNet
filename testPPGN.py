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

ppgn = PPGN.NoiselessJointPPGN(model, 2, 5, 8, verbose=2)
ppgn.compile(clf_metrics=['accuracy'])
ppgn.fit_classifier(x_train, y_train, validation_data=[x_test, y_test], epochs=5)

src, gen = ppgn.fit_gan(x_train, epochs=2000)

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
    img = np.concatenate((src[i], gen[i]), axis=1)*255
    img[img < 0  ] = 0
    img[img > 255] = 255
    cv2.imwrite('img/gan{}.bmp'.format(i), img)

h2_base = None
for i in range(10):
    samples, h2 = ppgn.sample(i, nbSamples=100, h2_start=h2_base)
    h2_base = h2[-1]
    img = np.concatenate((samples), axis=0)*255
    img[img < 0  ] = 0
    img[img > 255] = 255
    cv2.imwrite('img/samples{}.bmp'.format(i), img)

    










#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#
#
##Encoder 1 x-->Flatten
#enc1_input = Input(input_shape)
#enc1_output = enc1_input
#start=1
#stop=6
#for i in np.arange(start, stop, 1):
#    enc1_output = model.get_layer(index=i)(enc1_output)
#enc1 = Model(inputs=enc1_input, outputs=enc1_output)
#enc1.trainable=False
#
#
##Encoder 2 Flatten-->Dense 128
#enc2_input = Input((enc1.output_shape[1],))
#enc2_output = enc2_input
#start=6
#stop=8
#for i in np.arange(start, stop, 1):
#    enc2_output = model.get_layer(index=i)(enc2_output)
#enc2 = Model(inputs=enc2_input, outputs=enc2_output)
#enc2.trainable=False
#
#
##GAN-generator
#g_gen_input = Input((enc2.output_shape[1],))
#g_gen_output = g_gen_input
#g_gen_output = Dense(9216, activation='relu')(g_gen_output)
#g_gen_output = Reshape((12,12,64))(g_gen_output)
#g_gen_output = Dropout(0.25)(g_gen_output)
#g_gen_output = UpSampling2D((2,2))(g_gen_output)
#g_gen_output = Conv2DTranspose(32, (3,3), activation='relu')(g_gen_output)
#g_gen_output = Conv2DTranspose(1, (3,3), activation='sigmoid')(g_gen_output)
#g_gen = Model(inputs=g_gen_input, outputs=g_gen_output)
#
#
##GAN-discriminator
#g_disc_input = Input(input_shape)
#g_disc_output = g_disc_input
#g_disc_output = Conv2D(32, (3,3), activation='relu')(g_disc_output)
#g_disc_output = Conv2D(64, (3,3), activation='relu')(g_disc_output)
#g_disc_output = MaxPooling2D((2,2))(g_disc_output)
#g_disc_output = Dropout(0.25)(g_disc_output)
#g_disc_output = Flatten()(g_disc_output)
#g_disc_output = Dense(128, activation='relu')(g_disc_output)
#g_disc_output = Dropout(0.5)(g_disc_output)
#g_disc_output = Dense(2, activation='softmax')(g_disc_output)
#g_disc = Model(inputs=g_disc_input, outputs=g_disc_output)
#g_disc.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#g_disc.trainable=False
#
#
##Now the big Ugly, mega NET
#full_input = Input(input_shape)
##1st output
#full_output_img = full_input
#full_output_img = enc1(full_input)
#full_output_img = enc2(full_output_img)
#full_output_img = g_gen(full_output_img)
##2nd output
#full_output_disc = full_output_img
#full_output_disc = g_disc(full_output_disc)
##3rd output
#full_output_enc = full_output_img
#full_output_enc = enc1(full_output_enc)
##Create the model and compile
#full = Model(inputs=full_input, outputs=[full_output_img, full_output_disc, full_output_enc])
#full.compile(optimizer=keras.optimizers.Adadelta(),
#             metrics=None,
#             loss=[keras.losses.mean_squared_error, 
#                   keras.losses.categorical_crossentropy,
#                   keras.losses.mean_squared_error],
#             loss_weights=[1., 1., 1.])
#
#
#
#
#model.summary()
#enc1.summary()
#enc2.summary()
#g_gen.summary()
#g_disc.summary()
#full.summary()
#
#
##Train Step
##Classifier
#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test))
#
##Generate encoded version (enc1) of the dataset
#h_train = enc1.predict(x_train)
#h_test = enc1.predict(x_test)
#
##GAN
#report_freq = 500
#half_batch = 32
#g_epochs=30000
#disc_loss = []
#gen_loss = []
#for e in range(g_epochs):
#    #Train the discriminator
#    idX = np.random.randint(0, x_train.shape[0], half_batch)
#    
#    valid = x_train[idX]
#    fake = g_gen.predict(enc2.predict(h_train[idX]))
#    
#    v_ret = g_disc.train_on_batch(valid, keras.utils.to_categorical(np.ones ((half_batch)), 2))
#    f_ret = g_disc.test_on_batch(fake,   keras.utils.to_categorical(np.zeros((half_batch)), 2))
#    disc_loss.append(v_ret[0]*0.5 + f_ret[0]*0.5)
#    
#    #Train the GAN on a full batch
#    idX = np.random.randint(0, x_train.shape[0], 2*half_batch)
#    full_loss = full.train_on_batch(x_train[idX], [x_train[idX], 
#                                                   keras.utils.to_categorical(np.ones((2*half_batch)), 2),
#                                                   h_train[idX]])
#    gen_loss.append(full_loss)
#    
#    if e % report_freq == 0:
#        print(':::: Epoch #{} report ::::'.format(e))
#        print('GAN losses --- disc: {:.2f} // gen:{:.2f}'.format(disc_loss[-1], gen_loss[-1][2]))
#        print('Reconstruction losses --- img: {:.2f} // h1: {:.2f}'.format(gen_loss[-1][1], gen_loss[-1][3]))
#        
#        saveImg = np.concatenate((valid, fake), axis=2)
#        saveImg = np.vstack(saveImg)
#        saveImg = saveImg*255
#        saveImg[saveImg<0] = 0
#        saveImg[saveImg>255] = 255
#        cv2.imwrite('epoch_{}.bmp'.format(e), saveImg)
#        
#
##Sampling
##First create the sampler
#sampler_input = Input((g_gen.input_shape[1],))
#sampler_output = sampler_input
#sampler_output = g_gen(sampler_output)
#sampler_output = model(sampler_output)
#sampler = Model(inputs=sampler_input, outputs=sampler_output)
#sampler.trainable=False #Just in case
#sampler.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy) #We need to compile for the forward/backward pass
#
##For now take an arbitrary data point
#H = enc2.predict(enc1.predict(x_test[1:2]))
#Y = np.reshape(np.roll(y_test[1], 1), (1,10)) #And aim toward the next number
#
##This is a bit of keras/TF dark magic, bear with me
##Also see https://github.com/keras-team/keras/issues/2226
##We define a keras function that return the gradient after a Fwd/Bwd pass
#weights = sampler.weights
#grad = sampler.optimizer.get_gradients(sampler.total_loss, weights)
#input_tensors = [sampler.inputs[0],
#                 sampler.sample_weights[0],
#                 sampler.targets[0],
#                 K.learning_phase()]
#get_gradients = K.function(inputs=input_tensors, outputs=grad)
#
##Print the stating image
#img = g_gen.predict(H)
#img = img * 255
#img[img<0] = 0
#img[img>255] = 255
#img = img.astype('int')
#cv2.imwrite('sample_base.bmp', img[0])
#
##Sampling
#samples = 1e6
##espilon values are from paper
#eps1 = np.float64(1-3)
#eps2 = np.float64(1)
#eps3 = np.float64(1e-6)
#for s in range(samples):
#    #term1 is the reconstruction error
#    term1 = enc2.predict(enc1.predict(g_gen.predict(H)))
#    
#    #term2 is the gradient after a fwd/bwd pass
#    inputs = [H, [1], Y, 0] #[Sample, sample_weight, target, learning_phase] see input_tensors' def
#    #Take the gradient at input level. [0] refers to the first layer, hence the sum over all neuron in that layer
#    term2 = get_gradients(inputs)[0].sum(axis=1) 
#    term2 = np.reshape(term2, (1, term2.shape[0]))
#    
#    #term3 is just noise
#    term3 = np.random.normal(0, eps3**2, H.shape)
#    
#    H_old = H
#    H = H + eps1*term1 + eps2*term2 + term3
#    
#    if s %5e5 == 0:
#        img_old = img
#        img = g_gen.predict(H)
#        img = img * 255
#        img[img<0] = 0
#        img[img>255] = 255
#        img = img.astype('int')
#        cv2.imwrite('sample_{}.bmp'.format(s/1e5), img[0])
#        print('H diff: {:.2f}, img diff: {:.2f}'.format(np.abs(H-H_old).sum(), np.abs(img-img_old).sum()))
    
    

        
    
    






























