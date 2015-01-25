import sys
import signal
sys.path.append('..')
sys.path.append('../../data/')

import os, numpy as np
import scipy.io as sio
import time
import pdb
import color

import anglepy as ap
import anglepy.paramgraphics as paramgraphics
import anglepy.ndict as ndict

import theano
import theano.tensor as T
from collections import OrderedDict

import preprocessing as pp
import scipy.io as sio

toStr = np.vectorize(str)

def labelToMat(y):
    label = np.unique(y)
    newy = np.zeros((len(y), len(label)))
    for i in range(len(y)):
        newy[i, y[i]] = 1
    return newy.T
    
def infer1(data, v, layers=2):
  if True:
    res = [data]
    for i in xrange(layers):
      res += [np.log(1 + np.exp(v['w'+str(i)].dot(res[-1]) + v['b'+str(i)]))]
    for i in xrange(layers-1):
      res[i+2] = np.vstack((res[i+1], res[i+2]))
    return res[-1]
    
  else:
    res = data
    for i in xrange(layers):
      res = np.log(1 + np.exp(v['w'+str(i)].dot(res) + v['b'+str(i)]))
    return res
    
    
def main(n_z, n_hidden, dataset, seed, comment, alpha, decay1, decay2, gfx=True):
  
  # Initialize logdir
  import time
  logdir = 'results/gpulearn_mm_z_x_stacked_'+dataset+'_'+str(n_z)+'-'+'_'.join(toStr(n_hidden))+'_'+comment+'_'+str(int(time.time()))+'/'
  if not os.path.exists(logdir): os.makedirs(logdir)
  print 'logdir:', logdir
  print 'gpulearn_mm_z_x_stacked'
  color.printBlue('dataset = ' + str(dataset) + ' , n_z = ' + str(n_z) + ' , n_hidden = ' + str(n_hidden))
  with open(logdir+'hook.txt', 'a') as f:
    print >>f, 'learn_z_x', n_z, n_hidden, dataset, seed
  
  np.random.seed(seed)

  gfx_freq = 1
  
  weight_decay = 0
  
  # Init data
  if dataset == 'mnist':
    import anglepy.data.mnist as mnist
    
    # MNIST
    size = 28
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(size)
    f_enc, f_dec = pp.Identity()
    
    dir = 'models/mnist_z_x_50-500-500_longrun/'
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    train_x = infer1(train_x,v)
    test_x = infer1(test_x,v)
    valid_x = infer1(valid_x,v)

    # Test the features are right
    print test_x.shape
    print train_x.shape
    print valid_x.shape

    
    train_mean_prior = np.zeros((n_z,train_x.shape[1]))
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
    valid_mean_prior = np.zeros((n_z,valid_x.shape[1]))
    
    x = {'x': train_x.astype(np.float32), 'mean_prior': train_mean_prior.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'mean_prior': valid_mean_prior.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'mean_prior': test_mean_prior.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    
    L_valid = 1
    dim_input = (size,size)
    n_x = x_train['x'].shape[0]
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'gaussian'
    n_train = 50000
    n_test = 10000
    n_batch = 1000
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  elif dataset == 'mnist_rot': 
    # MNIST
    size = 28
    data_dir = '/home/lichongxuan/regbayes2/data/mat_data/'+'mnist_all_rotation_normalized_float_'
    tmp = sio.loadmat(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    
    dir = 'models/mnist_rot_z_x_50-500-500_longrun/'
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    train_x = infer1(train_x,v)
    test_x = infer1(test_x,v)
    valid_x = infer1(valid_x,v)
    '''
    # Test the features are right
    print test_x.shape
    print train_x.shape
    print valid_x.shape
    sio.savemat(logdir+'latent.mat', {'z_test': test_x, 'z_train': train_x})
    exit()
    '''
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    train_mean_prior = np.zeros((n_z,train_x.shape[1]))
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
    valid_mean_prior = np.zeros((n_z,valid_x.shape[1]))
    '''
    x = {'x': train_x.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    '''
    x = {'x': train_x.astype(np.float32), 'mean_prior': train_mean_prior.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'mean_prior': valid_mean_prior.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'mean_prior': test_mean_prior.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = x_train['x'].shape[0]
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'gaussian'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  elif dataset == 'mnist_back_rand': 
    # MNIST
    size = 28
    data_dir = '/home/lichongxuan/regbayes2/data/mat_data/'+'mnist_background_random_'
    tmp = sio.loadmat(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    
    dir = 'models/mnist_back_rand_z_x_50-500-500_longrun/'
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    train_x = infer1(train_x,v)
    test_x = infer1(test_x,v)
    valid_x = infer1(valid_x,v)
    '''
    # Test the features are right
    print test_x.shape
    print train_x.shape
    print valid_x.shape
    sio.savemat(logdir+'latent.mat', {'z_test': test_x, 'z_train': train_x})
    exit()
    '''
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    train_mean_prior = np.zeros((n_z,train_x.shape[1]))
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
    valid_mean_prior = np.zeros((n_z,valid_x.shape[1]))
    '''
    x = {'x': train_x.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    '''
    x = {'x': train_x.astype(np.float32), 'mean_prior': train_mean_prior.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'mean_prior': valid_mean_prior.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'mean_prior': test_mean_prior.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = x_train['x'].shape[0]
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'gaussian'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  elif dataset == 'mnist_back_image': 
    # MNIST
    size = 28
    data_dir = '/home/lichongxuan/regbayes2/data/mat_data/'+'mnist_background_images_'
    tmp = sio.loadmat(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    
    dir = 'models/mnist_back_image_z_x_50-500-500_longrun/'
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    train_x = infer1(train_x,v)
    test_x = infer1(test_x,v)
    valid_x = infer1(valid_x,v)
    '''
    # Test the features are right
    print test_x.shape
    print train_x.shape
    print valid_x.shape
    sio.savemat(logdir+'latent.mat', {'z_test': test_x, 'z_train': train_x})
    exit()
    '''
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    train_mean_prior = np.zeros((n_z,train_x.shape[1]))
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
    valid_mean_prior = np.zeros((n_z,valid_x.shape[1]))
    '''
    x = {'x': train_x.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    '''
    x = {'x': train_x.astype(np.float32), 'mean_prior': train_mean_prior.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'mean_prior': valid_mean_prior.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'mean_prior': test_mean_prior.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = x_train['x'].shape[0]
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'gaussian'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  elif dataset == 'mnist_back_image_rot': 
    # MNIST
    size = 28
    data_dir = '/home/lichongxuan/regbayes2/data/mat_data/'+'mnist_all_background_images_rotation_normalized_'
    tmp = sio.loadmat(data_dir+'train.mat')
    train_x = tmp['x_train'].T
    train_y = tmp['t_train'].T.astype(np.int32)
    # validation 2000
    valid_x = train_x[:,10000:]
    valid_y = train_y[10000:]
    train_x = train_x[:,:10000]
    train_y = train_y[:10000]
    tmp = sio.loadmat(data_dir+'test.mat')
    test_x = tmp['x_test'].T
    test_y = tmp['t_test'].T.astype(np.int32)
    
    dir = 'models/mnist_back_image_rot_z_x_50-500-500_longrun/'
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    train_x = infer1(train_x,v)
    test_x = infer1(test_x,v)
    valid_x = infer1(valid_x,v)
    '''
    # Test the features are right
    print test_x.shape
    print train_x.shape
    print valid_x.shape
    sio.savemat(logdir+'latent.mat', {'z_test': test_x, 'z_train': train_x})
    exit()
    '''
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    train_mean_prior = np.zeros((n_z,train_x.shape[1]))
    test_mean_prior = np.zeros((n_z,test_x.shape[1]))
    valid_mean_prior = np.zeros((n_z,valid_x.shape[1]))
    '''
    x = {'x': train_x.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    '''
    x = {'x': train_x.astype(np.float32), 'mean_prior': train_mean_prior.astype(np.float32), 'y': labelToMat(train_y).astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'mean_prior': valid_mean_prior.astype(np.float32), 'y': labelToMat(valid_y).astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'mean_prior': test_mean_prior.astype(np.float32), 'y': labelToMat(test_y).astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = x_train['x'].shape[0]
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'gaussian'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  elif dataset == 'norb':  
    import anglepy.data.norb as norb
    size = _size #48
    train_x, train_y, test_x, test_y = norb.load_resized(size, binarize_y=True)
    _x = {'x': train_x, 'y': train_y}
    ndict.shuffleCols(_x)
    train_x = _x['x']
    train_y = _x['y']
    
    # Do PCA
    f_enc, f_dec, pca_params = pp.PCA(_x['x'][:,:10000], cutoff=2000, toFloat=False)
    ndict.savez(pca_params, logdir+'pca_params')
    
    x = {'x': f_enc(train_x).astype(np.float32), 'y':train_y.astype(np.float32)}
    x_valid = {'x': f_enc(test_x).astype(np.float32), 'y':test_y.astype(np.float32)}
    x_test = {'x': f_enc(test_x).astype(np.float32), 'y':test_y.astype(np.float32)}
    
    L_valid = 1
    n_x = x['x'].shape[0]
    n_y = 5
    dim_input = (size,size)
    n_batch = 1000 #23400/900 = 27
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    mosaic_w = 5
    mosaic_h = 1
    type_px = 'gaussian'
  
  elif dataset == 'norb_pca':  
    # small NORB dataset
    import anglepy.data.norb as norb
    size = 48
    train_x, train_y, test_x, test_y = norb.load_resized(size, binarize_y=True)

    f_enc, f_dec, _ = pp.PCA(train_x, 0.999)
    #f_enc, f_dec, _ = pp.normalize_random(train_x)
    train_x = f_enc(train_x)
    test_x = f_enc(test_x)
    
    x = {'x': train_x.astype(np.float32)}
    x_valid = {'x': test_x.astype(np.float32)}
    L_valid = 1
    n_x = train_x.shape[0]
    dim_input = (size,size)
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'gaussian'
    nonlinear = 'softplus'
    n_batch = 900 #23400/900 = 27
    colorImg = False
    #binarize = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay= float(n_batch)/train_x.shape[1]

    
  elif dataset == 'svhn':
    # SVHN dataset
    import anglepy.data.svhn as svhn
    size = 32
    train_x, train_y, test_x, test_y = svhn.load_numpy(False, binarize_y=True) #norb.load_resized(size, binarize_y=True)
    extra_x, extra_y = svhn.load_numpy_extra(False, binarize_y=True)
    x = {'x': np.hstack((train_x, extra_x)), 'y':np.hstack((train_y, extra_y))}
    ndict.shuffleCols(x)
    
    #f_enc, f_dec, (x_sd, x_mean) = pp.preprocess_normalize01(train_x, True)
    f_enc, f_dec, pca_params = pp.PCA(x['x'][:,:10000], cutoff=1000, toFloat=True)
    ndict.savez(pca_params, logdir+'pca_params')
    
    n_y = 10
    x = {'x': f_enc(x['x']).astype(np.float32), 'y': x['y'].astype(np.float32)}
    x_valid = {'x': f_enc(test_x).astype(np.float32), 'y': test_y.astype(np.float32)}
    x_test = {'x': f_enc(test_x).astype(np.float32), 'y': test_y.astype(np.float32)}
    L_valid = 1
    n_x = x['x'].shape[0]
    dim_input = (size,size)
    n_batch = 5000
    colorImg = True
    bernoulli_x = False
    byteToFloat = False
    mosaic_w = 5
    mosaic_h = 2
    type_px = 'gaussian'
  
    
  # Construct model
  from anglepy.models import GPUVAE_MM_Z_X
  
  updates = get_adam_optimizer(learning_rate=alpha,decay1=decay1, decay2=decay2, weight_decay=weight_decay)
  model = GPUVAE_MM_Z_X(updates, n_x, n_y, n_hidden, n_z, n_hidden[::-1], nonlinear, nonlinear, type_px, type_qz=type_qz, type_pz=type_pz, prior_sd=100, init_sd=1e-3)
  
  if os.environ.has_key('pretrain') and bool(int(os.environ['pretrain'])) == True:
    #dir = '/Users/dpkingma/results/learn_z_x_mnist_binarized_50-(500, 500)_mog_1412689061/'
    #dir = '/Users/dpkingma/results/learn_z_x_svhn_bernoulli_300-(1000, 1000)_l1l2_sharing_and_1000HU_1412676966/'
    #dir = '/Users/dpkingma/results/learn_z_x_svhn_bernoulli_300-(1000, 1000)_l1l2_sharing_and_1000HU_1412695481/'
    #dir = '/Users/dpkingma/results/learn_z_x_mnist_binarized_50-(500, 500)_mog_1412695455/'
    #dir = '/Users/dpkingma/results/gpulearn_z_x_svhn_pca_300-(500, 500)__1413904756/'
    color.printBlue('pre-training')
    if dataset == 'mnist':
      dir = 'models/mnist_z_x_50-500-500_stack_1000_longrun/'
    elif dataset == 'mnist_rot':
      dir = 'models/mnist_rot_z_x_50-500-500_stack_1000_longrun/'
    elif dataset == 'mnist_back_rand':
      dir = 'models/mnist_back_rand_z_x_50-500-500_stack_1000_longrun/'
    elif dataset == 'mnist_back_image':
      dir = 'models/mnist_back_image_z_x_50-500-500_stack_1000_longrun/'
    elif dataset == 'mnist_back_image_rot':
      dir = 'models/mnist_back_image_rot_z_x_50-500-500_stack_1000_longrun/'
    elif dataset == 'svhn':
      dir = 'models/svhn_z_x_pca_300-500-500/'
    w = ndict.loadz(dir+'w_best.ndict.tar.gz')
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    ndict.set_value2(model.w, w)
    ndict.set_value2(model.v, v)
  
  # Some statistics for optimization
  ll_valid_stats = [-1e99, 0, 0]
  predy_valid_stats = [1, 0, 0, 0]
  predy_test_stats = [0, 1, 0]
  
  # Progress hook
  def hook(epoch, t, ll):
    
    if epoch%10 != 0: return
    
    
    ll_valid, _ = model.est_loglik(x_valid, n_samples=L_valid, n_batch=n_batch, byteToFloat=byteToFloat)
    
    # Log
    ndict.savez(ndict.get_value(model.v), logdir+'v')
    ndict.savez(ndict.get_value(model.w), logdir+'w')
    
    if ll_valid > ll_valid_stats[0]:
      ll_valid_stats[0] = ll_valid
      ll_valid_stats[1] = 0
      ll_valid_stats[2] = epoch
      ndict.savez(ndict.get_value(model.v), logdir+'v_best')
      ndict.savez(ndict.get_value(model.w), logdir+'w_best')
    else:
      ll_valid_stats[1] += 1
      # Stop when not improving validation set performance in 100 iterations
      if ll_valid_stats[1] > 1000:
        print "Finished"
        with open(logdir+'hook.txt', 'a') as f:
          print >>f, "Finished"
        exit()
    
    # Graphics
    if gfx and epoch%gfx_freq == 0:
      
      #tail = '.png'
      tail = '-'+str(epoch)+'.png'
      
      v = {i: model.v[i].get_value() for i in model.v}
      w = {i: model.w[i].get_value() for i in model.w}
        
      if 'pca' not in dataset and 'random' not in dataset and 'normalized' not in dataset:
        
        '''
        if 'w0' in v:
          image = paramgraphics.mat_to_img(f_dec(v['w0'][:].T), dim_input, True, colorImg=colorImg)
          image.save(logdir+'q_w0'+tail, 'PNG')
        
        image = paramgraphics.mat_to_img(f_dec(w['out_w'][:]), dim_input, True, colorImg=colorImg)
        image.save(logdir+'out_w'+tail, 'PNG')
        
        if 'out_unif' in w:
          image = paramgraphics.mat_to_img(f_dec(w['out_unif'].reshape((-1,1))), dim_input, True, colorImg=colorImg)
          image.save(logdir+'out_unif'+tail, 'PNG')
        '''
        if n_z == 2:
          n_width = 10
          import scipy.stats
          z = {'z':np.zeros((2,n_width**2))}
          for i in range(0,n_width):
            for j in range(0,n_width):
              z['z'][0,n_width*i+j] = scipy.stats.norm.ppf(float(i)/n_width+0.5/n_width)
              z['z'][1,n_width*i+j] = scipy.stats.norm.ppf(float(j)/n_width+0.5/n_width)
          
          x, _, _z = model.gen_xz({}, z, n_width**2)
          if dataset == 'mnist':
            x = 1 - _z['x']
          image = paramgraphics.mat_to_img(f_dec(_z['x']), dim_input)
          image.save(logdir+'2dmanifold'+tail, 'PNG')
        else:
          _x, _, _z_confab = model.gen_xz({}, {}, n_batch=144)
          x_samples = _z_confab['x']
          '''
          image = paramgraphics.mat_to_img(f_dec(x_samples), dim_input, colorImg=colorImg)
          image.save(logdir+'samples-prior'+tail, 'PNG')
          '''
          
          def infer(data, n_batch=1000):
            size = data['x'].shape[1]
            res = np.zeros((sum(n_hidden), size))
            predy = []
            for i in range(0, size, n_batch):
              idx_to = min(size, i+n_batch)
              x_batch = ndict.getCols(data, i, idx_to)
              _x, _z, _z_confab = model.gen_xz(x_batch, {}, n_batch)
              x_samples = _z_confab['x']
              
              
              predy += list(_z_confab['predy'])
              
              if i == -1:
                if epoch == 1:
                  print '_x'
                  for (d, x) in _x.items():
                    print d
                    print x.shape
                    
                  print '_z'
                  for (d, x) in _z.items():
                    print d
                    print x.shape
                    
                  print '_z_confab'
                  for (d, x) in _z_confab.items():
                    print d
                    
                      
                      
              for (hi, hidden) in enumerate(_z_confab['hidden']):
                res[sum(n_hidden[:hi]):sum(n_hidden[:hi+1]),i:i+n_batch] = hidden
            stats = dict()
            
            if epoch == -1:
              print 'features: ', res.shape
            
            return (res, predy, _z)

          def evaluate(data, predy):
            y = np.argmax(data['y'], axis=0)
            return sum([int(yi != py) for (yi, py) in zip(y, predy)]) / float(len(predy))

          (z_test, pred_test,_z_test) = infer(x_test)
          (z_valid, pred_valid,_z_valid) = infer(x_valid)
          (z_train, pred_train, _z_train) = infer(x_train)
          
          pre_valid = evaluate(x_valid, pred_valid)
          pre_test = evaluate(x_test, pred_test)
          
          if pre_valid < predy_valid_stats[0]:
            predy_valid_stats[0] = pre_valid
            predy_valid_stats[1] = pre_test
            predy_valid_stats[2] = epoch
            predy_valid_stats[3] = 0
          
            ndict.savez(ndict.get_value(model.v), logdir+'v_best_predy')
            ndict.savez(ndict.get_value(model.w), logdir+'w_best_predy')
          else:
            predy_valid_stats[3] += 1
            # Stop when not improving validation set performance in 100 iterations
            if predy_valid_stats[3] > 10000 and model.param_c.get_value() > 0:
              print "Finished"
              with open(logdir+'hook.txt', 'a') as f:
                print >>f, "Finished"
              exit()
          if pre_test < predy_test_stats[1]:
            predy_test_stats[0] = pre_valid
            predy_test_stats[1] = pre_test
            predy_test_stats[2] = epoch
          
          
          print 'c = ', model.param_c.get_value()
          print 'epoch', epoch, 't', t, 'll', ll, 'll_valid', ll_valid, 'valid_stats', ll_valid_stats
          print 'train_err = ', evaluate(x_train, pred_train), 'valid_err = ', evaluate(x_valid, pred_valid), 'test_err = ', evaluate(x_test, pred_test)
          print '--best: predy_valid_stats', predy_valid_stats, 'predy_test_stats', predy_test_stats
          with open(logdir+'hook.txt', 'a') as f:
            print >>f, 'epoch', epoch, 't', t, 'll', ll, 'll_valid', ll_valid, ll_valid_stats
            print >>f, 'train_err = ', evaluate(x_train, pred_train), 'valid_err = ', evaluate(x_valid, pred_valid), 'test_err = ', evaluate(x_test, pred_test)
          sio.savemat(logdir+'latent.mat', {'z_test': z_test, 'z_train': z_train})
        
        
        
        
        
          #x_samples = _x['x']
          #image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg)
          #image.save(logdir+'samples2'+tail, 'PNG')
          
      else:
        # Model with preprocessing
        
        if 'w0' in v:
          image = paramgraphics.mat_to_img(f_dec(v['w0'][:].T), dim_input, True, colorImg=colorImg)
          image.save(logdir+'q_w0'+tail, 'PNG')
          
        image = paramgraphics.mat_to_img(f_dec(w['out_w'][:]), dim_input, True, colorImg=colorImg)
        image.save(logdir+'out_w'+tail, 'PNG')

        _x, _z, _z_confab = model.gen_xz({}, {}, n_batch=144)
        x_samples = f_dec(_z_confab['x'])
        x_samples = np.minimum(np.maximum(x_samples, 0), 1)
        image = paramgraphics.mat_to_img(x_samples, dim_input, colorImg=colorImg)
        image.save(logdir+'samples'+tail, 'PNG')

  # Optimize
  #SFO
  dostep = epoch_vae_adam(model, x, n_batch=n_batch, bernoulli_x=bernoulli_x, byteToFloat=byteToFloat)
  loop_va(model, dostep, hook)
  
  pass

# Training loop for variational autoencoder
def loop_va(model, doEpoch, hook, n_epochs=10001):
  
  t0 = time.time()
  ct = 1000
  if os.environ.has_key('ct'):
    ct = int(os.environ['ct'])
  for t in xrange(1, n_epochs):
    if t >= ct:
      model.param_c.set_value(model.c)
    L = doEpoch()
    hook(t, time.time() - t0, L)
    
  print 'Optimization loop finished'

# Learning step for variational auto-encoder
def epoch_vae_adam(model, x, n_batch=100, convertImgs=False, bernoulli_x=False, byteToFloat=False):
  print 'Variational Auto-Encoder', n_batch
  
  def doEpoch():
    
    from collections import OrderedDict
    
    n_tot = x['x'].shape[1]
    idx_from = 0
    L = 0
    scores = []
    while idx_from < n_tot:
      idx_to = min(n_tot, idx_from+n_batch)
      x_minibatch = ndict.getCols(x, idx_from, idx_to)
      idx_from += n_batch
      
      
      if byteToFloat: x_minibatch['x'] = x_minibatch['x'].astype(np.float32)/256.
      if bernoulli_x: x_minibatch['x'] = np.random.binomial(n=1, p=x_minibatch['x']).astype(np.float32)
      
      
      # Do gradient ascent step
      L += model.evalAndUpdate(x_minibatch, {}).sum()
      #model.profmode.print_summary()
      
    L /= n_tot
    
    return L
    
  return doEpoch


def get_adam_optimizer(learning_rate=0.001, decay1=0.1, decay2=0.001, weight_decay=0.0):
  print 'AdaM', learning_rate, decay1, decay2, weight_decay
  def shared32(x, name=None, borrow=False):
    return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

  def get_optimizer(w, g):
    updates = OrderedDict()
    
    it = shared32(0.)
    updates[it] = it + 1.
    
    fix1 = 1.-(1.-decay1)**(it+1.) # To make estimates unbiased
    fix2 = 1.-(1.-decay2)**(it+1.) # To make estimates unbiased
    lr_t = learning_rate * T.sqrt(fix2) / fix1
    
    for i in w:
      gi = g[i]
      if weight_decay > 0:
        gi -= weight_decay * w[i] #T.tanh(w[i])

      # mean_squared_grad := E[g^2]_{t-1}
      mom1 = shared32(w[i].get_value() * 0.)
      mom2 = shared32(w[i].get_value() * 0.)
      
      # Update moments
      mom1_new = mom1 + decay1 * (gi - mom1)
      mom2_new = mom2 + decay2 * (T.sqr(gi) - mom2)
      
      # Compute the effective gradient and effective learning rate
      effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-10)
      
      effstep_new = lr_t * effgrad
      
      #print 'learning rate: ', lr_t.eval()
      
      # Do update
      w_new = w[i] + effstep_new
        
      # Apply update
      updates[w[i]] = w_new
      updates[mom1] = mom1_new
      updates[mom2] = mom2_new
      
    return updates
  
  return get_optimizer

#gfx = True
#n_z=int(sys.argv[2])
#n_hidden = tuple([int(sys.argv[3])]*int(sys.argv[4]))
#main(dataset=sys.argv[1], n_z=n_z, n_hidden=n_hidden, seed=0, comment=sys.argv[5])

