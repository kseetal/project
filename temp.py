from fcn import *

n_classes = 2          # Total number of classes
shape = ( 32,32,3 )    # Image Shape

x = fcn8( n_classes , shape )
model = x.get_model()

# plot the keras model in file 'model.jpg' by

visualize( model , 'model.jpg' )