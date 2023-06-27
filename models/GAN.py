import numpy
from keras.layers import Input,Conv2D,Conv2DTranspose,Flatten,Dense,BatchNormalization,Dropout,Lambda
from keras.layers import ReLU,LeakyReLU,Activation,UpSampling2D,ZeroPadding2D,Reshape
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import RandomNormal

class GAN:
    def __init__(self, input_size, discr_conv_filters, discr_activation, discr_dropout_rate, discr_lr,
                 gen_conv_filters, gen_activation, gen_dropout_rate, gen_lr, latent_dim):
        self.name = 'GAN'
        self.input_size = input_size
        self.discr_conv_filters = discr_conv_filters
        self.discr_layers = len(discr_conv_filters)
        self.discr_activation = discr_activation
        self.discr_dropout_rate = discr_dropout_rate
        self.discr_lr = discr_lr
        self.gen_conv_filters = gen_conv_filters
        self.gen_layers = len(gen_conv_filters)
        self.gen_activation = gen_activation
        self.gen_dropout_rate = gen_dropout_rate
        self.gen_lr = gen_lr
        self.latent_dim = latent_dim

        self.weight_initialization = RandomNormal(mean=0.0, std_dev = 0.02)

        self.disr_loss = []
        self.gen_loss = []

        self.discr_()
        self.gen_()
        self.adversarial_()

    def discr_(self):
        discr_input = Input(shape=self.input_dim, name = 'Discriminator_Input_Layer')
        x = discr_input

        for i in range(self.discr_layers):
            x = Conv2D(filters=self.discr_conv_filters[i],
                       kernel_size=3,strides=2,padding='same',
                       name = 'discriminator_conv_layer_'+str(i),
                       kernel_initializer=self.weight_initialization)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout()(x)
        
        x = Flatten()(x)

        discr_output = Dense(1,activation='sigmoid',kernel_initializer=self.weight_initialization)(x)

        self.discriminator = Model(discr_input,discr_output,name='Discriminator Model')


