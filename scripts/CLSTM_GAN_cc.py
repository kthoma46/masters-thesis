from __future__ import print_function, division
from keras.layers.advanced_activations import LeakyReLU

FILE_NAME = ''
TIME_DIM = 15
ACTIVATION = LeakyReLU(alpha=0.2)
LEARNING_RATE = 0.00005
TRAIN_SPLIT = 0.8

import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from keras.layers import ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Dropout, BatchNormalization
from keras.layers import Conv2D, Conv3D, ConvLSTM2D, DepthwiseConv2D, Dense
from keras.layers import Concatenate, MaxPooling1D, UpSampling2D, Activation
from keras.layers import DepthwiseConv2D, MaxPooling3D, MaxPooling1D, RepeatVector
from keras.layers import Lambda, Dot, Multiply, Input, Permute, LSTM, Dense, Conv2DTranspose
from keras.layers import Conv1D, Conv2D, ConvLSTM2D, Flatten, Reshape, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
import keras.backend as K

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split



class CLSTM_GAN():
    def __init__(self):
        self.img_rows = 20
        self.img_cols = 20
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = (TIME_DIM, 20, 20, 1)
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=LEARNING_RATE)

        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim))
        img = self.generator(z)
        self.critic.trainable = False
        valid = self.critic(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['mse'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred) #+ K.mean(abs(y_true - y_pred))

    def build_generator(self):
        noise = Input(shape=(TIME_DIM,20,20,1), name='INPUT')
        conv1 = ConvLSTM2D(32, 3, strides=1, activation=ACTIVATION, padding='same', return_sequences=True, name='CONV_1')(noise)
        bn1 = BatchNormalization(name='BN_1')(conv1)
        conv2 = ConvLSTM2D(64, 3, strides=1, activation=ACTIVATION, padding='same', return_sequences=True, name='CONV_2')(bn1)
        bn2 = BatchNormalization(name='BN_2')(conv2)
        conv3 = ConvLSTM2D(128, 3, strides=1, activation=ACTIVATION, padding='same', return_sequences=True, name='CONV_3')(bn2)
        bn3 = BatchNormalization(name='BN_3')(conv3)
        conv4 = ConvLSTM2D(256, 3, strides=1, activation=ACTIVATION, padding='same', return_sequences=True, name='CONV_4')(bn3)
        bn31 = BatchNormalization(name='BN_31')(conv4)

        merg = Concatenate()([bn1, bn2])
        merg2 = Concatenate()([bn3, bn31])
        merg3 = Concatenate()([merg, merg2])

        
        conv5 = Conv3D(1, 1, activation = 'softmax', name='CONV_5')(merg3)
        att = Multiply(name='ATTENTION')([conv5,conv4])
        mean = Lambda(lambda x: K.mean(x, axis=1), name='MEAN')(att)

        convt1 = Conv2D(512,3, strides=2, name='CONV_6')(mean)
        bn4 = BatchNormalization(name='BN_4')(convt1)
        act1 = Activation(ACTIVATION)(bn4)
        drop1 = Dropout(0.3, name='DROPOUT_1')(act1)
        convt2 = Conv2D(256,2, strides=2, name='CONV_7')(drop1)
        bn5 = BatchNormalization(name='BN_5')(convt2)
        act2 = Activation(ACTIVATION)(bn5)
        drop2 = Dropout(0.3, name='DROPOUT_2')(act2)
        convt3 = Conv2D(128,2, strides=2, name='CONV_8')(drop2)
        bn6 = BatchNormalization(name='BN_6')(convt3)
        act3 = Activation(ACTIVATION)(bn6)
        drop3 = Dropout(0.3, name='DROPOUT_3')(act3)
        convt4 = Conv2D(64,2, strides=2, name='CONV_9')(drop3)
        bn7 = BatchNormalization(name='BN_7')(convt4)
        act4 = Activation(ACTIVATION)(bn7)
        drop4 = Dropout(0.3, name='DROPOUT_4')(act4)
        #convt5 = Conv2D(64,2, strides=2, name='CONV_10')(drop4)
        #bn8 = BatchNormalization(name='BN_8')(convt5)
        #act5 = Activation(ACTIVATION)(bn8)
        #drop5 = Dropout(0.3, name='DROPOUT_5')(act5)
        flat = Flatten()(drop4)
        fc = Dense(100, activation=ACTIVATION)(flat)
        final = Dense(1, activation='tanh')(fc)
        
        model = Model(inputs=noise,outputs=final)
        model.summary()
        return model

    def build_critic(self):
        model = Sequential()
        model.add(Dense(1,input_shape=(1,), activation=ACTIVATION))
        model.add(Dense(1, activation=ACTIVATION))
        model.add(Dense(1, activation=ACTIVATION))
        model.add(Dense(1, activation='tanh'))
        model.summary()
        img = Input(shape=(1,))
        validity = model(img)
        return Model(img, validity)

    def data_loader(self):
        data = np.load(FILE_NAME)
        data = (data.astype(np.float32) - .5) / .5
        x=[]
        y=[]
        SIZE = data.shape[0]-TIME_DIM+1
        for i in range(SIZE-1):
            x.append(data[i:i+TIME_DIM,:,:,:])
            y.append(data[i+TIME_DIM,:,:,:])
        x = np.array(x)
        y = np.array(y)
        loc = int(TRAIN_SPLIT*x.shape[0]) 
        X_train = x[:loc]
        X_test = x[loc:]
        Y_train = y[:loc]
        Y_test = y[loc:]        
        return X_train,Y_train,X_test,Y_test


    def train(self, epochs, batch_size=8, sample_interval=10):
        noise, X_train, noise2, X_test = self.data_loader()
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                gen_imgs = self.generator.predict(noise[idx])
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            g_loss = self.combined.train_on_batch(noise[idx], valid)
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            if epoch % sample_interval == 0:
                self.sample_images(epoch, noise2)

if __name__ == '__main__':
    wgan = CLSTM_GAN()
    wgan.train(epochs=10001, batch_size=10000, sample_interval=1000)