import sys
sys.path.append('..')
sys.path.append('../../data/')

import os, numpy as np
import scipy.io as sio
import time

import anglepy as ap
import anglepy.paramgraphics as paramgraphics
import anglepy.ndict as ndict

import theano
import theano.tensor as T
from collections import OrderedDict

import preprocessing as pp
import color

def main(n_z, n_hidden, dataset, seed, comment, gfx=True):
  
  # Initialize logdir
  import time
  if os.environ.has_key('pretrain') and bool(int(os.environ['pretrain'])) == True:
    comment+='_pre-train'
  if os.environ.has_key('prior') and bool(int(os.environ['prior'])) == True:
    comment+='_prior'
  if os.environ.has_key('train_residual') and bool(int(os.environ['train_residual'])) == True:
    comment+='_train-residual'
  if os.environ.has_key('sigma_square'):
    comment+=('_'+str(float(os.environ['sigma_square'])))
  logdir = 'results/gpulearn_z_x_'+dataset+'_'+str(n_z)+'-'+str(n_hidden)+comment+'_'+str(int(time.time()))+'/'
  if not os.path.exists(logdir): os.makedirs(logdir)
  print 'logdir:', logdir
  print 'gpulearn_z_x', n_z, n_hidden, dataset, seed
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
    
    if os.environ.has_key('prior') and bool(int(os.environ['prior'])) == True:
        color.printBlue('Loading prior')
        mnist_prior = sio.loadmat('data/mnist_prior/mnist_prior.mat')
        train_mean_prior = mnist_prior['z_train']
        test_mean_prior = mnist_prior['z_test']
        valid_mean_prior = mnist_prior['z_valid']
    else:
        train_mean_prior = np.zeros((n_z,train_x.shape[1]))
        test_mean_prior = np.zeros((n_z,test_x.shape[1]))
        valid_mean_prior = np.zeros((n_z,valid_x.shape[1]))
    
    
    x = {'x': train_x.astype(np.float32), 'mean_prior': train_mean_prior.astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32), 'mean_prior': valid_mean_prior.astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32), 'mean_prior': test_mean_prior.astype(np.float32)}
    
    L_valid = 1
    dim_input = (size,size)
    n_x = size*size
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'bernoulli'
    n_train = 50000
    n_test = 10000
    n_batch = 1000
    colorImg = False
    bernoulli_x = True
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
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    x = {'x': train_x.astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = size*size
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'bernoulli'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = True
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
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    x = {'x': train_x.astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = size*size
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'bernoulli'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = True
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
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    x = {'x': train_x.astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = size*size
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'bernoulli'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = True
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
    
    print train_x.shape
    print train_y.shape
    print test_x.shape
    print test_y.shape
    
    f_enc, f_dec = pp.Identity()
    x = {'x': train_x.astype(np.float32)}
    x_train = x
    x_valid = {'x': valid_x.astype(np.float32)}
    x_test = {'x': test_x.astype(np.float32)}
    L_valid = 1
    dim_input = (size,size)
    n_x = size*size
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    nonlinear = 'softplus'
    type_px = 'bernoulli'
    n_train = 12000
    n_test = 50000
    n_batch = 240
    colorImg = False
    bernoulli_x = True
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  if dataset == 'mnist_binarized':
    import anglepy.data.mnist_binarized as mnist_binarized
    # MNIST
    train_x, valid_x, test_x = mnist_binarized.load_numpy(28)
    x = {'x': np.hstack((train_x, valid_x)).astype(np.float32)}
    x_valid = {'x': test_x.astype(np.float32)}
    L_valid = 1
    dim_input = (28,28)
    n_x = 28*28
    n_y = 10
    type_qz = 'gaussianmarg'
    type_pz = 'mog'
    nonlinear = 'rectlin'
    type_px = 'bernoulli'
    n_train = 60000
    n_batch = 1000
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train
    
  elif dataset == 'freyface':
    # Frey's face
    import anglepy.data.freyface as freyface
    n_train = 1600
    train_x = freyface.load_numpy()
    np.random.shuffle(train_x)
    x = {'x': train_x.T[:,0:n_train]}
    x_valid = {'x': train_x.T[:,n_train:]}
    L_valid = 1
    dim_input = (28,20)
    n_x = 20*28
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'bounded01'
    nonlinear = 'tanh'  #tanh works better with freyface #'softplus'
    n_batch = 100
    colorImg = False
    bernoulli_x = False
    byteToFloat = False
    weight_decay = float(n_batch)/n_train

  elif dataset == 'freyface_pca':
    # Frey's face
    import anglepy.data.freyface as freyface
    n_train = 1600
    train_x = freyface.load_numpy().T
    np.random.shuffle(train_x.T)
    
    f_enc, f_dec, _ = pp.PCA(train_x, 0.99)
    train_x = f_enc(train_x)
    
    x = {'x': train_x[:,0:n_train].astype(np.float32)}
    x_valid = {'x': train_x[:,n_train:].astype(np.float32)}
    L_valid = 1
    dim_input = (28,20)
    n_x = train_x.shape[0]
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'gaussian'
    nonlinear = 'softplus'
    n_batch = 100
    colorImg = False
    bernoulli_x = False
    byteToFloat = False

  elif dataset == 'freyface_bernoulli':
    # Frey's face
    import anglepy.data.freyface as freyface
    n_train = 1600
    train_x = freyface.load_numpy().T
    np.random.shuffle(train_x.T)
    
    x = {'x': train_x[:,0:n_train].astype(np.float32)}
    x_valid = {'x': train_x[:,n_train:].astype(np.float32)}
    L_valid = 1
    dim_input = (28,20)
    n_x = train_x.shape[0]
    type_pz = 'gaussianmarg'
    type_px = 'bernoulli'
    nonlinear = 'softplus'
    n_batch = 100
    colorImg = False
    bernoulli_x = False
    byteToFloat = False

  elif dataset == 'norb':  
    # small NORB dataset
    import anglepy.data.norb as norb
    size = 48
    train_x, train_y, test_x, test_y = norb.load_resized(size, binarize_y=True)

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
    byteToFloat = False
    bernoulli_x = False
    weight_decay= float(n_batch)/train_x.shape[1]
  
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

  elif dataset == 'norb_normalized':
    # small NORB dataset
    import anglepy.data.norb as norb
    size = 48
    train_x, train_y, test_x, test_y = norb.load_resized(size, binarize_y=True)

    #f_enc, f_dec, _ = pp.PCA(train_x, 0.99)
    #f_enc, f_dec, _ = pp.normalize_random(train_x)
    f_enc, f_dec, _ = pp.normalize(train_x)
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
    
    print 'Performing PCA, can take a few minutes... ',
    f_enc, f_dec, pca_params = pp.PCA(x['x'][:,:10000], cutoff=600, toFloat=True)
    ndict.savez(pca_params, logdir+'pca_params')
    print 'Done.'
    
    n_y = 10
    x = {'x': f_enc(x['x']).astype(np.float32)}
    x_valid = {'x': f_enc(test_x).astype(np.float32)}
    L_valid = 1
    n_x = x['x'].shape[0]
    dim_input = (size,size)
    n_batch = 5000
    colorImg = True
    bernoulli_x = False
    byteToFloat = False
    type_qz = 'gaussianmarg'
    type_pz = 'gaussianmarg'
    type_px = 'gaussian'
    nonlinear = 'softplus'
  
    
  # Construct model
  from anglepy.models import GPUVAE_Z_X
  updates = get_adam_optimizer(learning_rate=3e-4, weight_decay=weight_decay)
  model = GPUVAE_Z_X(updates, n_x, n_hidden, n_z, n_hidden[::-1], nonlinear, nonlinear, type_px, type_qz=type_qz, type_pz=type_pz, prior_sd=100, init_sd=1e-3)
  
  if False:
    #dir = '/Users/dpkingma/results/learn_z_x_mnist_binarized_50-(500, 500)_mog_1412689061/'
    #dir = '/Users/dpkingma/results/learn_z_x_svhn_bernoulli_300-(1000, 1000)_l1l2_sharing_and_1000HU_1412676966/'
    #dir = '/Users/dpkingma/results/learn_z_x_svhn_bernoulli_300-(1000, 1000)_l1l2_sharing_and_1000HU_1412695481/'
    #dir = '/Users/dpkingma/results/learn_z_x_mnist_binarized_50-(500, 500)_mog_1412695455/'
    #dir = '/Users/dpkingma/results/gpulearn_z_x_svhn_pca_300-(500, 500)__1413904756/'
    dir = '/home/ubuntu/results/gpulearn_z_x_mnist_50-[500, 500]__1414259423/'
    w = ndict.loadz(dir+'w_best.ndict.tar.gz')
    v = ndict.loadz(dir+'v_best.ndict.tar.gz')
    ndict.set_value(model.w, w)
    ndict.set_value(model.v, v)
  
  # Some statistics for optimization
  ll_valid_stats = [-1e99, 0]
  
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
    
    print epoch, t, ll, ll_valid, ll_valid_stats
    with open(logdir+'hook.txt', 'a') as f:
      print >>f, epoch, t, ll, ll_valid, ll_valid_stats

    # Graphics
    if gfx and epoch%gfx_freq == 0:
      
      #tail = '.png'
      tail = '-'+str(epoch)+'.png'
      
      v = {i: model.v[i].get_value() for i in model.v}
      w = {i: model.w[i].get_value() for i in model.w}
        
      if 'pca' not in dataset and 'random' not in dataset and 'normalized' not in dataset:
        
        if 'w0' in v:
          image = paramgraphics.mat_to_img(f_dec(v['w0'][:].T), dim_input, True, colorImg=colorImg)
          image.save(logdir+'q_w0'+tail, 'PNG')
        
        image = paramgraphics.mat_to_img(f_dec(w['out_w'][:]), dim_input, True, colorImg=colorImg)
        image.save(logdir+'out_w'+tail, 'PNG')
        
        if 'out_unif' in w:
          image = paramgraphics.mat_to_img(f_dec(w['out_unif'].reshape((-1,1))), dim_input, True, colorImg=colorImg)
          image.save(logdir+'out_unif'+tail, 'PNG')
        
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
          if not(os.environ.has_key('train_residual') and bool(int(os.environ['train_residual'])) == True) and (os.environ.has_key('prior') and bool(int(os.environ['prior'])) == True):
            nn_batch_nn = 144
            mp_in = np.random.randint(0,x_train['mean_prior'].shape[1],nn_batch_nn)
            m_p = x_train['mean_prior'][:,mp_in]
            s_s = 1
            if os.environ.has_key('sigma_square'):
                s_s = float(os.environ['sigma_square'])
            x_samples = model.gen_xz_prior({}, {}, m_p, s_s, n_batch=144)
            x_samples = x_samples['x']
            m_p1 = (np.ones((n_z, nn_batch_nn)).T * np.mean(x_train['mean_prior'], axis = 1)).T
            x_samples1 = model.gen_xz_prior({}, {}, m_p1.astype(np.float32), s_s, n_batch=144)
            image = paramgraphics.mat_to_img(f_dec(x_samples1['x']), dim_input, colorImg=colorImg)
            image.save(logdir+'mean_samples-prior'+tail, 'PNG')
            x_samples11 = model.gen_xz_prior11({}, {}, m_p, s_s, n_batch=144)
            image = paramgraphics.mat_to_img(f_dec(x_samples11['x']), dim_input, colorImg=colorImg)
            image.save(logdir+'prior-image'+tail, 'PNG')
          else:
            _x, _, _z_confab = model.gen_xz({}, {}, n_batch=144)
            x_samples = _z_confab['x']
          image = paramgraphics.mat_to_img(f_dec(x_samples), dim_input, colorImg=colorImg)
          image.save(logdir+'samples-prior'+tail, 'PNG')
          
          def infer(data, n_batch=1000):
            size = data['x'].shape[1]
            res = np.zeros((sum(n_hidden), size))
            res1 = np.zeros((n_z,size))
            res2 = np.zeros((n_hidden[-1],size))
            res3 = np.zeros((n_z,size))
            for i in range(0, size, n_batch):
              idx_to = min(size, i+n_batch)
              x_batch = ndict.getCols(data, i, idx_to)
              _x, _z, _z_confab = model.gen_xz(x_batch, {}, n_batch)
              x_samples = _z_confab['x']
              for (hi, hidden) in enumerate(_z_confab['hidden']):
                res[sum(n_hidden[:hi]):sum(n_hidden[:hi+1]),i:i+n_batch] = hidden
              res1[:,i:i+n_batch] = _z_confab['mean']
              res2[:,i:i+n_batch] = _z_confab['hidden'][-1]
              res3[:,i:i+n_batch] = _z_confab['logvar']
            return res, res1, res2, res3
          
          z_test, z_test1, z_test2, vv_test = infer(x_test)
          z_train, z_train1, z_train2, vv_train = infer(x_train)
          
          l_t, px_t, pz_t, qz_t = model.test(x_train, n_samples=1, n_batch=n_batch, byteToFloat=byteToFloat)
          print 'Elogpx', px_t, 'Elogpz', pz_t, '-Elogqz', qz_t
          #sigma_square = float(os.environ['sigma_square'])
          print 'var', np.mean(np.exp(vv_train)), 'q', np.mean(np.abs(z_train1)), 'p', np.mean(np.abs(train_mean_prior)), 'd', np.mean(np.abs(z_train1-train_mean_prior))
          with open(logdir+'hook.txt', 'a') as f:
            print >>f, 'Elogpx', px_t, 'Elogpz', pz_t, '-Elogqz', qz_t
            print >>f, 'var', np.mean(np.exp(vv_train)), 'q', np.mean(np.abs(z_train1)), 'p', np.mean(np.abs(train_mean_prior)), 'd', np.mean(np.abs(z_train1-train_mean_prior))
          
          
          sio.savemat(logdir+'full_latent.mat', {'z_test': z_test, 'z_train': z_train})
          sio.savemat(logdir+'mean_latent.mat', {'z_test': z_test1, 'z_train': z_train1})
          sio.savemat(logdir+'last_latent.mat', {'z_test': z_test2, 'z_train': z_train2})

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
  loop_va(dostep, hook)
  
  pass

# Training loop for variational autoencoder
def loop_va(doEpoch, hook, n_epochs=10001):
  
  t0 = time.time()
  for t in xrange(1, n_epochs):
    L = doEpoch()
    hook(t, time.time() - t0, L)
    
  print 'Optimization loop finished'

# Learning step for variational auto-encoder
def epoch_vae_adam(model, x, n_batch=100, convertImgs=False, bernoulli_x=False, byteToFloat=False):
  print 'Variational Auto-Encoder', n_batch
  
  def doEpoch():
    
    from collections import OrderedDict

    n_tot = x.itervalues().next().shape[1]
    idx_from = 0
    L = 0
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

