import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from time import strftime, time
import matplotlib.pyplot as plt
import utils

class SoundSurrogate(Model):
    def __init__(self, x_in, f_in, model_path=None):
        super(SoundSurrogate, self).__init__()

        self.x, self.f = x_in, f_in

        self.model = self._network_(C=self.x.shape[-1])
        self.build_model(x_in)
        self.build(input_shape=self.x.shape)
        if model_path is not None:
            self.load_weights(model_path)
            
        self.optimizer = tf.keras.optimizers.Adam()

    def _network_(self, C, k=4):
        name = ['SoundSurrogate']

        layer = tf.keras.Sequential()
        if self.x is not None:
            layer.add(self.x)

        C_list = [16, 32, 64]
        for i, C_i in enumerate(C_list):
            name.append('layer{0:02d}'.format(i))
            layer.add(tf.keras.layers.Dense(C_i, name='/'.join(name)))
            layer.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            name.pop()

        name.append('layer_out')
        layer.add(tf.keras.layers.Dense(int(self.f.shape[1]), name='/'.join(name)))
        layer.add(tf.keras.layers.ReLU())
        name.pop()

        return layer

    def call(self, x):
        return self.model(x)

    def run_model(self, x):
        return self.model(x)

    def build_model(self, x):
        xx = tf.random.normal([1, x.shape[-1]], dtype=tf.dtypes.float32)
        _ = self.run_model(xx)

    def get_layers(self):
        idx, layer_list = 0, []
        try:
            while True:
                layer_list.append(self.layers[0].get_layer(index=idx))
                idx += 1
        except:
            pass

        N_params = 0
        for l in layer_list:
            w = l.get_weights()
            for w_i in w:
                N_params += w_i.size
                

        print('Number of params: {}'.format(N_params))
        
        return layer_list

    def compute_loss(self, x, f, reporting=False):
        f_out = self.run_model(x)
        
        loss = 1000.*tf.reduce_mean((f_out - f)**2)
        
        return loss

    def train_step(self, x, f):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, f)
            
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def compute_dataset_loss(self, ds):
        loss, N = 0., 0
        for x_i, f_i in ds:
            N_batch = x_i.shape[0]

            l_batch = self.compute_loss(x_i, f_i)

            loss += l_batch
            N += N_batch

        return loss/N

    def fit(self, ds_train, ds_test=None, N_epochs=100, batch_size=100, learning_rate=1e-4, decay_rate=0.99, 
                  save_every=1, print_every=10, save_model_path=None, verbose=True):

        iters = 0
        for epoch in range(1, N_epochs+1):
            if verbose and ((epoch == 1) or (epoch % print_every) == 0):
                print('Epoch: '+str(epoch))
                print('Learning Rate: {:.9f}'.format(self.optimizer.learning_rate.numpy()))
            start_time = time()

            for x_i, f_i in ds_train:
                iters += 1
                N_batch = x_i.shape[0]

                self.train_step(x_i, f_i)
                
            if verbose and ((epoch == 1) or (epoch % print_every) == 0):
                train_loss = self.compute_dataset_loss(ds_train)

                print('-- Training --')
                print('L = {0:04f}, '.format(train_loss))

                if ds_test is not None:
                    test_loss = self.compute_dataset_loss(ds_test)

                    print('-- Testing --')
                    print('L = {0:04f}, '.format(test_loss))
                print('Epoch took %.2f seconds\n' %(time() - start_time), flush=True)

            self.optimizer.learning_rate = decay_rate*self.optimizer.learning_rate

            # Save Model
            if save_model_path is not None and ((epoch % save_every) == 0):
                model_epoch = save_model_path+'/{0:08d}'.format(epoch)
                if not os.path.exists(model_epoch):
                    os.makedirs(model_epoch)
                self.save_weights('/'.join([model_epoch, 'aeroNN.h5']), save_format='h5')

        if save_model_path is not None:
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            self.save_weights('/'.join([save_model_path, 'aeroNN.h5']), save_format='h5')







