############################
#
#   Ugo Louche - 2018
#
############################
## Based on PPGN's paper by Nguyen et al. (2017)
## A (noiseless joint) PPGN network (henceforth PPGN for short) has two main components:
## - A classifier network (e.g. performing image classification)
## - A GAN network (generator + discriminator)
##
## The entire network is made of a succession of two autoencoders with recontructions
## targets set up to two hidden layers in the classifier network. 
## The gist of the PPGN method is, to provide a way to generate synthetic
## input through a sampling procedure such as said input maximize the output
## of a given neuron anywhere in the classifier network. 
## 
## The classifier and GAN networks might be pretrained, or require fine-tuning.
## Freeze your layers accordingly before passing them and/or call the relevant 'fit'
## methods provided in this class
##
## Main parameters are:
##  - classifier: the classifier network to use
##  - hidden1_index: the index of the classifier network's layer's OUTPUT used as the first AE target
##  - hidden2_index: the index of the classifier network's layer's OUTPUT used as the second AE target
##  - output_index:  the index of the classifier network's layer's OUTPUT where the maximized neuron
##                   is located during the sampling procedure
##  - gan_generator (opt) : the generator part of the GAN network
##  - gan_discriminator (opt) : the discriminator part of the GAN network
##
## Note that even if no training will be performed, a call to the 'compile' method is required
## for two reasons: 1) compile actually build all the interconnections between the two main networks
## and 2) the sampling procedure perform a forward backward pass to compute a gradient, hence some compilation
## (in the keras' sense) must be performed. 
##
## There are four level of verbose. Roughly speaking, level 0 print nothing, level 1 logs events on
## the two main networks (classifier and GAN) and level 2 logs events on the internal network
## (enc1, enc2, g_gen, g_disc, full and sampler). level 3 prints models summary() in keras style
## on top of all of the above
## The verbose output can also be redirected to a file via the log_path parameter.
##

import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D, Conv2DTranspose,  Reshape

class NoiselessJointPPGN:
    nextID = 0 #Static variable for identification and count
    
    def __init(self, classifier, hidden1_index, hidden2_index, output_index,
               gan_generator='Default', gan_discriminator='Default',
               verbose=3, log_path=None, name=None): 
        self.name = 'PPGN_{}'.format(NoiselessJointPPGN.netxID) if name is None else name
        NoiselessJointPPGN.nextID+=1
        
        self.verbose=verbose
        self.log_path=log_path
        self._log('Creating PPGN #{}'.format(NoiselessJointPPGN.nextID-1), 1)
        self._log('verbose mode: {}'.format(self.verbose), 1)
        
        self._isCompiled=False
        self._classifierSet=False
        self._ganSet=False
        
        self.set_classifier(classifier, hidden1_index, hidden2_index, output_index)
        
        self.set_GAN(gan_generator, gan_discriminator)
        
        return
    
    
    #A LOT of parameters here, 
    def compile(self,
                # Loss for the classifier network
                clf_loss=keras.losses.categorical_crossentropy,\
                # Optimizer for the classifier network
                clf_opti=keras.optimizers.Adadelta(),\
                # Metrics for the classifier network                  
                clf_metrics=None,\
                # Loss for the GAN (discriminator) network                                        
                g_disc_loss=keras.losses.binary_crossentropy,\
                # Optimizer  for the GAN (discriminator) network
                g_disc_opti=keras.optimizers.Adadelta(),\
                # Metrics for the GAN (discriminator) network
                g_disc_metrics=None,\
                # Losses for the GAN (generator) network                                   
                gan_loss=[keras.losses.mean_squared_error,\
                          keras.losses.binary_crossentropy,\
                          keras.losses.mean_squared_error],\
                # Losses' weights  for the GAN (generator) network
                gan_loss_weight=[1., 1., 1.],\
                # Optimizer for the GAN (generator) network                         
                gan_opti=keras.optimizers.Adadelta(),\
                # Loss for the sampler network                 
                sampler_loss=keras.losses.categorical_crossentropy,\
                # Optimizer for the sampler network
                sampler_opti=keras.optimizers.Adadelta()\
                ):
        if not self._classifierSet and self._ganSet:
            self._log('Cannot compile PPGN without setting classifier and GAN first', 0)
            return
        
        ## Create the various subnetworks and set training flags accordingly and compile them if needed
        # Classifier network is already set, just make sur it is trainable nd compile
        self.classifier.trainable=True
        self.classifier.compile(loss=clf_loss, optimizer=clf_opti, metrics=clf_metrics)
        
        self._log('Classifier Network compiled', 2)
        
        #Enc1 is the first subset of classifier and is not trainable
        enc1_input = Input(self.input_shape)
        enc1_output = enc1_input
        for i in np.arange(1, self.h1_ind+1, 1):
            enc1_output = self.classifier.get_layer(index=i)(enc1_output)
        self.enc1 = Model(inputs=enc1_input, outputs=enc1_output)
        self.enc1.trainable = False
        
        self._log('Created enc1 (not trainable) network from layer #{} to layer #{} (included)'\
                  .format(1, self.h1_ind), 2)
        self._log('with input shape: {}, output_shape: {}'.\
                  format(self.enc1.input_shape[1:], self.enc1.output_shape[1:]), 2)
        self._print_summary(self.enc1, 'Encoder 1')
        
        #Enc2 is the second subset of classifier and is not trainable
        enc2_input = Input(self.enc1.output_shape[1:])
        enc2_output = enc2_input
        for i in np.arange(self.h1_ind+1, self.h2_ind+1, 1):
            enc2_output = self.classifier.get_layer(index=i)(enc2_output)
        self.enc2 = Model(inputs=enc2_input, outputs=enc2_output)
        self.enc2.trainable = False
        
        self._log('Created enc2 network (not trainable) from layer #{} to layer #{} (included)'\
                  .format(self.h1_ind+1, self.h2_ind), 2)
        self._log('with input shape: {}, output_shape: {}'.\
                  format(self.enc2.input_shape[1:], self.enc2.output_shape[1:]), 2)
        self._print_summary(self.enc1, 'Encoder 1')
        
        #Compile g_disc as trainable
        self.g_disc.trainable=True
        self.g_disc.compile(loss=g_disc_loss, optimizer=g_disc_opti, metrics=g_disc_metrics)
        
        self._log('g_disc network compiled as trainable', 2)
        
        #Freeze g_disc, create the full (generative) GAN network and compile it
        self.g_disc.trainable=False
        gan_input = Input(self.input_shape)
        #1st output
        gan_output_img = gan_input
        gan_output_img = self.enc1(gan_output_img)
        gan_output_img = self.enc2(gan_output_img)
        gan_output_img = self.g_gen(gan_output_img)
        #2nd output
        gan_output_disc = self.g_disc(gan_output_img)
        #3rd output
        gan_output_enc = self.enc1(gan_output_img)
        self.gan = Model(inputs=gan_input, outputs=[gan_output_img, gan_output_disc, gan_output_enc])
        self.gan.trainable = True
        self.gan.compile(loss=gan_loss, loss_weights=gan_loss_weight, optimizer=gan_opti)
        
        self._log('Created gan network (trainable) from enc1, enc2 g_disc and g_gen', 2)
        self._log('with input shape: {} and output shapes: image->{}, adversarial(disc)->{}, h1(encoder)->{}'\
                  .format(self.gan.output_shape[0][1:], self.gan.output_shape[1][1:], self.gan.output_shape[2][1:]), 2)
        self._print_summary(self.gan, 'Full GAN')
        
        #Create the sampler network, freeze it and compile
        sampler_input = Input(self.h2_shape)
        sampler_output = sampler_input
        sampler_output = self.g_gen(sampler_output)
        sampler_output = self.classifier(sampler_output)
        self.sampler = Model(inputs=sampler_input, outputs=sampler_output)
        self.sampler.trainable = False
        self.sampler.compile(loss=sampler_loss, optimizer=sampler_opti)
        
        self._log('Created sampler network (not trainable) from g_gen and classifier', 2)
        self._log('with input shape: {] and output shape: {}'\
                  .format(self.sampler.input_shape[1:], self.sampler.output_shape[1:]), 2)
        self._print_summary(self.sampler, 'sampler')
        
        self._isCompiled=True       
        return
    
    
    def set_classifier(self, classifier, hidden1_index, hidden2_index, output_index):
        self.input_shape = classifier.input_shape[1:]
        
        self.h1_ind = hidden1_index
        self.h1_shape = classifier.get_layer(index=hidden1_index).output_shape[1:]
        
        self.h2_ind = hidden2_index
        self.h2_shape = classifier.get_layer(index=hidden2_index).output_shape[1:]
        
        self.out_ind = output_index
        self.out_shape = classifier.get_layer(index=output_index).output_shape[1:]
        
        self.classifier=classifier
        self._classifierSet=True
        #Invalidate GAN and compilation flag
        self._ganSet=False
        self._isCompiled=False
        
        self._log('Classifier network set -- input shape: {}, output_shape: {}'\
                  .format(self.input_shape, self.classifier.output_shape[1:]), 1)
        self._log('hidden 1 -- index={}, shape={}'.format(self.h1_ind, self.h1_shape), 1)
        self._log('hidden 2 -- index={}, shape={}'.format(self.h2_ind, self.h2_shape), 1)
        self._log('output -- index={}, shape={}'.format(self.out_ind, self.out_shape), 1)
        self._print_summary(self.classifier, 'Classifier')
        
        return
    
    
    def set_GAN(self, gan_generator, gan_discriminator):
        self.g_gen = self._defaultGANgenerator()      if gan_generator=='Default'     else gan_generator
        self.g_disc = self._defaultGANdiscriminator() if gan_discriminator=='Default' else gan_discriminator
        self._ganSet=True
        #Invalidate Compilation
        self._isCompiled=False
        
        self._log('GAN network set !', 1)
        self._log('g_gen -- input_shape: {}, output_shape: {}'\
                  .format(self.g_gen.input_shape[1:], self.g_gen.output_shape[:1]), 2)
        self._print_summary(self.g_gen, 'GAN-generator')
        self._log('g_disc -- input_shape: {}, output_shape: {}'\
                  .format(self.g_disc.input_shape[1:], self.g_disc.output_shape[:1]), 2)
        self._print_summary(self.g_disc, 'GAN-discriminator')
    
    
    def fit_classifier(self):
        return
    
    
    def fit_gan(self):
        return
    
    
    def sample(self):
        return
    
    
    def _defaultGANgenerator(self):
        if not self._classifierSet:
            self._log('Classifier is not set. Cannot create a default GAN-generator', 0)
            return None
        
        #Placeholder, use something that actually LOOKS LIKE a GAN's generator
        g_gen_input = Input(self.h2_shape)
        g_gen_output = g_gen_input
        g_gen_output = Dense(9216, activation='relu')(g_gen_output)
        g_gen_output = Reshape((12,12,64))(g_gen_output)
        g_gen_output = Dropout(0.25)(g_gen_output)
        g_gen_output = UpSampling2D((2,2))(g_gen_output)
        g_gen_output = Conv2DTranspose(32, (3,3), activation='relu')(g_gen_output)
        g_gen_output = Conv2DTranspose(1, (3,3), activation='tanh')(g_gen_output)
        g_gen = Model(inputs=g_gen_input, outputs=g_gen_output)
        
        return g_gen
    
    
    def _defaultGANdiscriminator(self):
        if not self._classifierSet:
            self._log('Classifier is not set. Cannot create a default GAN-generator', 0)
            return None
        
        #Placeholder, use something that actually LOOKS LIKE a GAN's discriminator
        g_disc_input = Input(self.input_shape)
        g_disc_output = g_disc_input
        g_disc_output = Conv2D(32, (3,3), activation='relu')(g_disc_output)
        g_disc_output = Conv2D(64, (3,3), activation='relu')(g_disc_output)
        g_disc_output = MaxPooling2D((2,2))(g_disc_output)
        g_disc_output = Dropout(0.25)(g_disc_output)
        g_disc_output = Flatten()(g_disc_output)
        g_disc_output = Dense(128, activation='relu')(g_disc_output)
        g_disc_output = Dropout(0.5)(g_disc_output)
        g_disc_output = Dense(1, activation='sigmoid')(g_disc_output)
        g_disc = Model(inputs=g_disc_input, outputs=g_disc_output)
        
        return g_disc
    
    
    def _print_summary(self, model, name):
        self._log('Printing summary for '+name, 3)
        model.summary(print_fn=lambda x: self._log(x, 3))
        return
    
    
    def _log(self, text, level):
        text = self.name+":::"+text
        if level <= self.verbose: 
            if self.log_path is not None:
                f = open(self.log_path, 'w')
                f.write(text+'\n')
                f.close()
            else:
                print(text)
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    