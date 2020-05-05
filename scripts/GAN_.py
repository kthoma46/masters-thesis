from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from numpy.linalg import norm
from keras.losses import hinge, squared_hinge

import keras.backend as K
import matplotlib.pyplot as plt
import sys
import numpy as np

class GAN():
    def __init__(self):
        self.file_name = 'data/Cincinnati.npy'
        self.look_ahead = 0
        self.time_slice = 7
        self.img_rows = 50
        self.img_cols = 50
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = (self.img_rows, self.img_cols, self.time_slice)
        self.train_split = 0.8
        self.G_unit = [16,32,64,128,128,64,32,1]
        self.G_k_size = [3,3,3,3,3,3,3,3]
        self.G_stride = [1,1,1,1,1,1,1,1]
        self.G_dropout = [0.3,0.3,0.3,0.3,0.3,0.3,0.3]
        self.G_pool = [2,2]
        self.G_activation = LeakyReLU(alpha=0.2)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=self.latent_dim)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred) + squared_hinge(y_true, y_pred)#hinge(y_true, y_pred)#K.mean((y_true - y_pred)*(y_true - y_pred)) #K.mean(abs(y_true - y_pred))

    def build_generator(self):

        noise = Input(shape=self.latent_dim, name='INPUT')

        conv2d = Dense(7, activation = 'softmax', name='CONV_3D')(noise)
        att = Multiply(name='ATTENTION')([conv2d,noise])

        conv1 = Conv2D(self.G_unit[0], self.G_k_size[0], strides=self.G_stride[0], padding='same', name='CONV_1')(att)
        bn1 = BatchNormalization(name='BN_1')(conv1)
        act1 = Activation(self.G_activation, name='ACTIVATION_1')(bn1)
        drop1 = Dropout(self.G_dropout[0], name='DROPOUT_1')(act1)

        conv2 = Conv2D(self.G_unit[1], self.G_k_size[1], strides=self.G_stride[1], padding='same', name='CONV_2')(drop1)
        bn2 = BatchNormalization(name='BN_2')(conv2)
        act2 = Activation(self.G_activation, name='ACTIVATION_2')(bn2)
        pool1 = MaxPooling2D(self.G_pool[0], name = 'POOL_1')(act2)
        drop2 = Dropout(self.G_dropout[1], name='DROPOUT_2')(pool1)

        conv3 = Conv2D(self.G_unit[2], self.G_k_size[2], strides=self.G_stride[2], padding='same', name='CONV_3')(drop2)
        bn3 = BatchNormalization(name='BN_3')(conv3)
        act3 = Activation(self.G_activation, name='ACTIVATION_3')(bn3)
        drop3 = Dropout(self.G_dropout[2], name='DROPOUT_3')(act3)

        conv4 = Conv2D(self.G_unit[3], self.G_k_size[3], strides=self.G_stride[3], padding='same', name='CONV_4')(drop3)
        bn4 = BatchNormalization(name='BN_4')(conv4)
        act4 = Activation(self.G_activation, name='ACTIVATION_4')(bn4)
        drop4 = Dropout(self.G_dropout[3], name='DROPOUT_4')(act4)

        conv5 = Conv2DTranspose(self.G_unit[4], self.G_k_size[4], strides=self.G_stride[4], padding='same', name='CONV_5')(drop4)
        bn5 = BatchNormalization(name='BN_5')(conv5)
        act5 = Activation(self.G_activation, name='ACTIVATION_5')(bn5)
        drop5 = Dropout(self.G_dropout[4], name='DROPOUT_5')(act5)

        conv6 = Conv2DTranspose(self.G_unit[5], self.G_k_size[5], strides=self.G_stride[5], padding='same', name='CONV_6')(drop5)                
        bn6 = BatchNormalization(name='BN_6')(conv6)
        act6 = Activation(self.G_activation, name='ACTIVATION_6')(bn6)
        usamp1 = UpSampling2D(self.G_pool[1], interpolation='nearest', name='UP_SAMPLING_1')(act6)
        drop6 = Dropout(self.G_dropout[5], name='DROPOUT_6')(usamp1)

        conv7 = Conv2DTranspose(self.G_unit[6], self.G_k_size[6], strides=self.G_stride[6], padding='same', name='CONV_7')(drop6)
        bn7 = BatchNormalization(name='BN_7')(conv7)
        act7 = Activation(self.G_activation, name='ACTIVATION_7')(bn7)
        drop7 = Dropout(self.G_dropout[6], name='DROPOUT_7')(act7)

        conv8 = Conv2DTranspose(self.G_unit[7], self.G_k_size[7], strides=self.G_stride[7], padding='same', name='CONV_8')(drop7) 
        bn8 = BatchNormalization(name='BN_8')(conv8)
        act8 = Activation('tanh', name='ACTIVATION_8')(bn8)               

        model = Model(inputs=noise, outputs=act8)

        model.summary()
        return model

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def data_scale(self, a):
        a = 2*a
        a = a-1
        return a

    def data_norm(self, mat):
        a = mat
        a = np.reshape(a,(a.shape[0],a.shape[1]*a.shape[2]*a.shape[3]))
        self.mu = np.mean(a,axis=0)
        self.variance = np.var(a,axis=0)

        a = (a-self.mu)/(self.variance+0.00001)
        a = np.reshape(a,(mat.shape[0],mat.shape[1],mat.shape[2],mat.shape[3]))
        return a

    def data_slicer(self, mat, lab):
        a = []
        b = []
        for i in range(mat.shape[0]-self.time_slice):
            a.append(mat[i:i+self.time_slice,:,:,:])
            b.append(lab[i+self.time_slice,:,:,:])
        a = np.array(a)
        a = np.reshape(a,(mat.shape[0]-self.time_slice,mat.shape[1],mat.shape[2],self.time_slice))
        b = np.array(b)
        b = np.reshape(b,(lab.shape[0]-self.time_slice,lab.shape[1],lab.shape[2],lab.shape[3]))
        return a, b

    def data_loader(self):
        data = np.load(self.file_name)
        data[data>0]=1
        x=[]
        y=[]        
        x = data[:data.shape[0]-self.look_ahead,:,:,:]
        y = data[self.look_ahead:data.shape[0],:,:,:]
        x = np.array(x)
        y = np.array(y)
        x, y = self.data_slicer(x,y)
        loc = int(self.train_split*x.shape[0]) 
        X_train = x[:loc]
        X_test = x[loc:]
        Y_train = y[:loc]
        Y_test = y[loc:]  
        X_train = self.data_norm(X_train)
        X_test = self.data_norm(X_test)   
        Y_train = self.data_scale(Y_train)
        Y_test = self.data_scale(Y_test)
        return X_train, Y_train, X_test, Y_test

    def train(self, epochs, batch_size=64, sample_interval=50):

        # Load the dataset
        noise, X_train, noise2, X_test = self.data_loader()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                # Sample noise as generator input
                #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))


                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise[idx])

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise[idx], valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, noise2, X_test)

    def sample_images(self, epoch, noise, X_test):
        r, c = 5, 2*5
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        #gen_imgs[gen_imgs>0.5]=1
        #gen_imgs[gen_imgs<=0.5]=0

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if j%2==0:
                    axs[i,j].imshow(X_test[cnt,:,:,0], cmap='gray')
                    axs[i,j].axis('off')
                else:
                    axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                    axs[i,j].axis('off')                
                cnt += 1
        fig.savefig("images/opioid_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=4000, batch_size=64, sample_interval=10)
