import numpy as np
from keras.layers import Input, Conv2D, Conv2DTranspose,Lambda, Flatten, Dense, Activation, LeakyReLU, BatchNormalization, Dropout, Reshape
from keras.models import Model
from keras.backend import int_shape,mean,square,random_normal,shape,exp
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop,Adadelta,Adagrad

class VariationalAutoEncoder():
    def __init__(self, input_dim, e_filters, e_kernel,e_strides,
                 d_t_filters, d_t_kernel, d_t_strides,
                 z_dim,
                 use_batch_norm = False, use_dropout = False):
                    self.name = 'Variational AutoEncoder'
                    self.input_dim = input_dim
                    self.e_filters = e_filters
                    self.e_kernel = e_kernel
                    self.e_strides = e_strides
                    self.d_t_filters = d_t_filters
                    self.d_t_kernel = self.d_t_kernel
                    self.d_t_strides = self.d_t_strides
                    self.z_dim = z_dim
                    self.use_batch_norm = use_batch_norm
                    self.use_dropout = use_dropout

                    self.num_encoder_layers = len(e_filters)
                    self.num_decoder_layers = len(d_t_filters)

                    self.build()
    
    def build(self):
            encoder_input_layer = Input(shape=(self.input_dim),name='encoder_input')
            x = encoder_input_layer
            for i in range(self.num_encoder_layers):
                    conv_layer = Conv2D(
                            filters=self.e_filters[i],
                            kernel_size=self.e_kernel[i],
                            strides=self.e_kernel[i],
                            padding='same',
                            name='encoder_conv_layer'+str(i)
                    )

            x = conv_layer(x)

            if self.use_batch_norm:
                    x = BatchNormalization()(x)
                    
            x = LeakyReLU()(x)

            if self.use_dropout:
                    x = Dropout()(x)

            x = LeakyReLU()(x)

            curr_shape = int_shape(x[1:])

            x = Flatten()(x)

            self.mu = Dense(self.z_dim,name='mu')(x)

            self.log_var = Dense(self.z_dim,name='log_var')(x)

            self.encoder_mu_log_var = Model(encoder_input_layer,(self.mu,self.log_var))

            def sampling(args):
                    mu, log_var = args
                    epsilon = random_normal(shape=shape(mu),mean=0,stddev=1.0)
                    return mu + exp(log_var/2)*epsilon
            
            encoder_output_layer = Lambda(sampling, name = 'encoder_output')([self.mu,self.log_var])

            self.encoder = Model(encoder_input_layer,encoder_output_layer)
            

            
    
        