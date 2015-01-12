import pylearn2.utils.serial as pio
import numpy as np
import scipy.io as sio

str = 'mnist_all_background_images_rotation_normalized_'
if False:
    str1 = 'train_valid'
    z = pio.load(str+str1+'.amat')
    x_train = z[:,:-1]
    t_train = z[:,-1]
    print x_train.shape
    print t_train.shape
    sio.savemat(str+str1+'.mat', {'x_train':x_train,'t_train':t_train})
else:
    str1 = 'test'
    z = pio.load(str+str1+'.amat')
    x_test = z[:,:-1]
    t_test = z[:,-1]
    print x_test.shape
    print t_test.shape
    sio.savemat(str+str1+'.mat', {'x_test':x_test,'t_test':t_test})

