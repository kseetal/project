

import keras
from keras.layers import Conv2D, Dropout, Conv2DTranspose, Add, Softmax
from keras.layers.core import Activation
from keras.layers.convolutional import Cropping2D
from keras.models import Model
from plot_model import plot_model


from encoder import vgg_encoder
from filter import bilinear


class fcn8( object ):


  def __init__(self, n_classes, shape ):
    self.n_classes = n_classes
    self.shape = shape

  def get_model(self):
  
    n_classes = self.n_classes
    shape = self.shape

    img_input , [f1 , f2 , f3 , f4 , f5 ] = vgg_encoder( shape )
  
    o = f5
    o = Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same' )(o)
    o = Dropout(0.5)(o)
    o = Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same' )(o)
    o = Dropout(0.5)(o)

    o = Conv2D( n_classes ,  ( 1 , 1 ) ,activation = 'relu' )(o)
    o = Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, kernel_initializer=bilinear )(o)
    o = Cropping2D(((1, 1), (1, 1)))(o)

    o2 = f4
    o2 = Conv2D( n_classes ,  ( 1 , 1 ) ,activation = 'relu')(o2)
	
    o = Add()([ o , o2 ])
    o = Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(o)
    o = Cropping2D(((1, 1), (1, 1)))(o)
  
    o2 = f3 
    o2 =Conv2D( n_classes ,  ( 1 , 1), activation = 'relu' )(o2)
  
    o  = Add()([ o2 , o ])
    o = Conv2DTranspose( n_classes , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False )(o)
    o = Cropping2D(((4, 4), (4, 4)))(o)
    o = Softmax(axis=3)(o)
	
    model = Model(img_input , o )
    model.model_name = "fcn_8"
    return model


def visualize(model,filename ):
  plot_model(model, to_file=filename,show_shapes=True)
