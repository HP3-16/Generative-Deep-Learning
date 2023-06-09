import numpy as np
from keras.layers import Input, Conv2D, Flatten, Dense,LeakyReLU, BatchNormalization, Dropout, Conv2DTranspose
from keras.models import Model
from keras.backend import int_shape
from keras.utils import plot_model
from keras.optimizers import Adam,RMSprop,Adadelta,Adagrad

class AutoEncoder():
    def __init__(self, input_dim, e_filters, e_kernel,e_strides,
                 d_t_filters, d_t_kernel, d_t_strides,
                 z_dim,
                 use_batch_norm = False, use_dropout = False
                 ):
                    self.name = 'AutoEncoder'
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
            encoder_input_layer = Input(shape=self.input_dim,name='Encoder_Input')
            x = encoder_input_layer
            for i in range(self.num_encoder_layers):
                    conv_layer = Conv2D(filters=self.e_filters[i],
                                        kernel_size=self.e_kernel[i],
                                        strides = self.e_strides[i],
                                        padding='same', name = 'encoder_conv_layer_'+str(i)
                                        )
                    x = conv_layer(x)
                    x = LeakyReLU()(x)

                    if self.use_batch_norm:
                            x = BatchNormalization()(x)
                    if self.use_dropout:
                            x = Dropout()(x)
                    
            curr_shape = int_shape(x)[1:]
            
            x = Flatten()(x)
            
            encoder_output_layer = Dense(self.z_dim,name='encoder_output')(x)

            self.encoder = Model(encoder_input_layer,encoder_output_layer)


