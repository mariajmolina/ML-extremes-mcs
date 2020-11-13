import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DLFrontStyle:
    """
    Class instantiation of DLFrontStyle.
    Build and train a convolutional neural network architecture based on 
    "Automated detection of weather fronts using a deep learning neural network" 
    by James C. Biard and Kenneth E. Kunkel.
    https://search.proquest.com/openview/0f35e404bb8d72ca9e34812d0b30345e/1?pq-origsite=gscholar&cbl=2037689
    
    Attributes:
        variable ([str]): Variable(s) string as a list.
        dim (int tuple): Tuple of data spatial dimensions. 
                         Defaults to ``(105, 161)`` for era5.
                         Choose ``(106,81)`` for ``003``.
        learning_rate (float): The training learning rate. Defaults to ``0.0001``.
        epochs (int): Number of epochs for training. Defaults to ``10``.
        conv1 (int): Number of feature maps in convolutional layer 1. Defaults to ``80``.
        conv2 (int): Number of feature maps in convolutional layer 2. Defaults to ``80``.
        conv3 (int): Number of feature maps in convolutional layer 3. Defaults to ``80``.
        acti1 (str): Activation function for convolutional layer 1. Defaults to ``relu``.
        acti2 (str): Activation function for convolutional layer 2. Defaults to ``relu``.
        acti3 (str): Activation function for convolutional layer 3. Defaults to ``relu``.
        k1_size (int): Size of kernel (filter) as tuple for layer 1. Defaults to ``(5,5)``.
        k2_size (int): Size of kernel (filter) as tuple for layer 2. Defaults to ``(5,5)``.
        k3_size (int): Size of kernel (filter) as tuple for layer 3. Defaults to ``(5,5)``.
        batch_norm (boolean): Whether to apply batch normalization after every convolutional layer. Defaults to ``True``.
        spatial_drop (float): Whether to apply spatial dropout after every convolutional layer. 
                              Defaults to dropout percentage ``0.3`` for 30%.
        output_shape (int): Number of output channels. Defaults to ``2``.
        output_activation (str): Activation function for last layer. Defaults to ``softmax`` (dlfront default and use 
                                 if output_shape==2). Other options include ``sigmoid`` if output_shape==1.
        loss_function (str): Loss function for training. Defaults to ``categorical_crossentropy``.
        verbose (int): Whether training silently (0), with progress bar (1), or one line per epoch (2). Defaults to 1.
        metrics (list str): Metric(s) for evaluating model performance as list. 
                            Defaults to ['accuracy', 'mean_squared_error', 'mean_absolute_error'].
        model_save (str): Directory to save trained model files. Defaults to ``None``.
        model_num (int): Model number for saving. Defaults to ``None``.
        
    """
    def __init__(self, variable, dim=(105, 161), 
                 learning_rate=0.0001, epochs=10, 
                 conv1=80, conv2=80, conv3=80,
                 acti1='relu', acti2='relu', acti3='relu',
                 k1_size=(5,5), k2_size=(5,5), k3_size=(5,5),
                 batch_norm=True, spatial_drop=0.3,
                 output_shape=2, output_activation='softmax', 
                 loss_function='categorical_crossentropy', verbose=1,
                 metrics=['accuracy', 'mean_squared_error', 'mean_absolute_error'],
                 model_save=None, model_num=None):
        
        self.variable = variable
        self.dim = dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.acti1 = acti1
        self.acti2 = acti2
        self.acti3 = acti3
        self.k1_size = k1_size
        self.k2_size = k2_size
        self.k3_size = k3_size
        self.batch_norm = batch_norm
        self.spatial_drop = spatial_drop
        self.output_shape = output_shape
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.verbose = verbose
        self.metrics = metrics
        self.model_save = model_save
        self.model_num = model_num
        
    def compile_model(self):
        """
        Build and compile the DL-front style convolutional neural network.
        Args:
            input_data (array): Input channels to reference for shape of network layers.
        """
        # generate input tensor
        inputs = keras.Input(shape=(*self.dim, len(self.variable,)))
        # generate first convolutional layer
        conv2d = layers.Conv2D(self.conv1, 
                               self.k1_size,
                               strides=(1,1),
                               padding='same', 
                               data_format='channels_last',
                               dilation_rate=1, activation=self.acti1, 
                               use_bias=True, 
                               kernel_initializer='glorot_uniform', 
                               bias_initializer='zeros', 
                               #kernel_regularizer=l2(0.001), bias_regularizer=None, 
                               activity_regularizer=None, 
                               kernel_constraint=None, 
                               bias_constraint=None)
        # pass input into first conv layer
        x = conv2d(inputs)
        # generate second conv layer and pass data
        x = layers.Conv2D(self.conv2, 
                               self.k2_size,
                               strides=(1,1),
                               padding='same', 
                               data_format='channels_last',
                               dilation_rate=1, activation=self.acti2, 
                               use_bias=True, 
                               kernel_initializer='glorot_uniform', 
                               bias_initializer='zeros', 
                               #kernel_regularizer=l2(0.001), bias_regularizer=None, 
                               activity_regularizer=None, 
                               kernel_constraint=None, 
                               bias_constraint=None)(x)
        # generate third conv layer and pass data
        x = layers.Conv2D(self.conv3, 
                               self.k3_size,
                               strides=(1,1),
                               padding='same', 
                               data_format='channels_last',
                               dilation_rate=1, activation=self.acti3, 
                               use_bias=True, 
                               kernel_initializer='glorot_uniform', 
                               bias_initializer='zeros', 
                               #kernel_regularizer=l2(0.001), bias_regularizer=None, 
                               activity_regularizer=None, 
                               kernel_constraint=None, 
                               bias_constraint=None)(x)
        # generate final layer and pass data
        outputs = layers.Conv2DTranspose(self.output_shape, 
                                           self.k3_size,
                                           strides=(1,1),
                                           padding='same', 
                                           data_format='channels_last',
                                           dilation_rate=1, activation=self.output_activation, 
                                           use_bias=True, 
                                           kernel_initializer='glorot_uniform', 
                                           bias_initializer='zeros', 
                                           #kernel_regularizer=l2(0.001), bias_regularizer=None, 
                                           activity_regularizer=None, 
                                           kernel_constraint=None, 
                                           bias_constraint=None)(x)
        # assemble model
        model = keras.Model(inputs=inputs, outputs=outputs, name="dlfront_style")
        # compile model
        model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.loss_function, 
                      metrics=self.metrics)
        # print model summary
        print(model.summary())
        #keras.utils.plot_model(model, show_shapes=True, dpi=600)
        return model
    
    def train_dl(self, model, generator):
        """
        Train the compiled DL model, save the trained model, and save the history and metric information from training to 
        ``self.dl_filedirectory``.
            
        Args: 
            model (keras.engine.sequential.Sequential): Compiled deep convolutional neural network.
            generator (DataGenerator(keras.utils.Sequence)): Keras data generator.
        """
        history=model.fit(generator, epochs=self.epochs, verbose=self.verbose, 
                                      ## use_multiprocessing=True,
                                      workers=0)
        if self.model_save:
            pd.DataFrame(history.history).to_csv(f'/{self.model_save}/model_{self.model_num}.csv')
            keras.models.save_model(model, f"/{self.model_save}/model_{self.model_num}.h5")
            if self.verbose == 1:
                return model
        if not self.model_save:
            return model
