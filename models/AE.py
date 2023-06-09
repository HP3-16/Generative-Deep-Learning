import numpy as np
from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Activation, LeakyReLU, BatchNormalization, Dropout, Reshape
from keras.models import Model
from keras.backend import int_shape,mean,square
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
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
            '''
            Builds the encoder and decoder of the AE model
            '''
    # Encoder        
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

    # Decoder
            decoder_input_layer = Input(shape=self.z_dim,name='decoder_input')

            x = Dense(np.prod(curr_shape))(decoder_input_layer)
            x = Reshape(curr_shape)(x)

            for i in range(self.num_decoder_layers):
                    conv_t_layer = Conv2DTranspose(filters=self.d_t_filters[i],
                                                   kernel_size=self.d_t_kernel[i],
                                                   strides=self.d_t_strides[i],
                                                   padding='same',
                                                   name = 'conv_layer_t_'+str(i)
                                                   )
                    x = conv_t_layer(x)

                    if i < self.num_decoder_layers - 1:
                            x = LeakyReLU()(x)

                            if self.use_batch_norm:
                                    x = BatchNormalization()(x)
                            
                            if self.use_dropout:
                                    x = Dropout()(x)
                    else:
                            x = Activation('sigmoid')
                    
            decoder_output_layer = x
            self.decoder = Model(decoder_input_layer,decoder_output_layer)

            model_input_layer = encoder_input_layer
            model_output_layer = self.decoder(encoder_output_layer)

            self.model = Model(model_input_layer,model_output_layer)

    def compile(self, eta):
            self.eta = eta
            
            Optimizer = Adam(learning_rate=eta)
            
            def RMSE_loss(y,y_hat):
                    '''
                    Here the loss is RMSE
                    '''
                    return mean(square(y - y_hat),axis=[1,2,3])
            
            self.model.compile(optimizer=Optimizer, loss=RMSE_loss)
