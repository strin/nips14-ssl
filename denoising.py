import numpy as np
import anglepy.ndict as ndict
import scipy.io as sio
import cPickle, gzip
import math

# load data, recognition model and generative model
print 'Loading data...'

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

dir = './models/mnist_z_x_50-500-500_longrun/'
v = ndict.loadz(dir+'v.ndict.tar.gz')
w = ndict.loadz(dir+'w.ndict.tar.gz')

# choose number of images to transform and number of images to do visualization
num_trans = 10000
num_show = 100
data = (x_test[:num_trans,:]).T
pertub_label = np.ones(data.shape)

# perturb data
print 'Perturbing data...'

width = 28
height = 28
pertub_type = 3
noise_type = 1 # 0 or uniformly random
denoise_tpye = 1 # sample or mean 

if pertub_type == 1:
    data_perturbed = data + np.random.normal(0,0.4,(data.shape))
elif pertub_type == 2:
    data_perturbed = data.copy()
    data_perturbed *= (np.random.random(data.shape) > 0.6)
elif pertub_type == 3:
    data_perturbed = data.copy()
    begin_h = 8
    begin_w = 8
    rec_h = 12
    rec_w = 12
    rectengle = np.zeros(rec_h*rec_w)
    for i in xrange(rec_h):
        rectengle[i*rec_w:(i+1)*rec_w]=np.arange((begin_h+i)*width+begin_w,(begin_h+i)*width+begin_w+rec_w)
    
    if noise_type == 1:
        data_perturbed[rectengle.astype(np.int32),:] = 0
    else:
        data_perturbed[rectengle.astype(np.int32),:] = np.random.random((rectengle.shape[0],data.shape[1]))
    
    pertub_label[rectengle.astype(np.int32),:] = 0
#elif pertub_type == 4:
#    data_perturbed = data.copy()
#    rec_h = 12
#    rec_w = 12
    
sio.savemat('noise_rawdata.mat', {'z_train' : x_train.T, 'z_test' : data_perturbed})

    
# denoise
print 'Denoising...'
n_hidden_q = 2
n_hidden_p = 2
denoise_times = 15 # denoising epoch

output = np.zeros(data.shape+(denoise_times+2,))
output[:,:,0] = data
output[:,:,1] = data_perturbed

z_train = x_train.copy().T
#z_train = np.ones((v['w'+str(n_hidden_q-1)].shape[0],num_trans))

for t in xrange(2,denoise_times+2):
    tmp = output[:,:,t-1]
    # sample z
    for i in range(n_hidden_q):
        tmp = np.log(1 + np.exp(v['w'+str(i)].dot(tmp) + v['b'+str(i)]))
        # save features for prediction
        if t == 2:
            z_train = np.log(1 + np.exp(v['w'+str(i)].dot(z_train) + v['b'+str(i)]))
            sio.savemat('noise_features.mat', {'z_train' : z_train, 'z_test' : tmp})
        if t == denoise_times+1:
            sio.savemat('de-noise_features.mat', {'z_train' : z_train, 'z_test' : tmp})
            
    q_mean = v['mean_w'].dot(tmp) + v['mean_b']
    
    if denoise_tpye == 1:
        q_logvar = v['logvar_w'].dot(tmp) + v['logvar_b']
        eps = np.random.normal(0, 1, (q_mean.shape))
        tmp = q_mean + np.exp(0.5*q_logvar) * eps
    elif denoise_tpye == 2:
        tmp = q_mean
        
    # generate x
    for i in range(n_hidden_p):
        tmp = np.log(1 + np.exp(w['w'+str(i)].dot(tmp) + w['b'+str(i)]))
    tmp = 1/(1 + np.exp(-(w['out_w'].dot(tmp)+w['out_b'])))
    
    output[:,:,t] = pertub_label*data+(1-pertub_label)*tmp
    
sio.savemat('de-noise_rawdata.mat', {'z_train' : x_train.T, 'z_test' : output[:,:,-1]})

# save data to do visualization
print 'Visualizing...'
visualization_image_number = num_show

# left a gap for sub-images
w = width+1
h = height+1

# layout the sub-images on a big image
w_image_number = int(math.sqrt(visualization_image_number) + 1)
h_image_number = int((visualization_image_number + w_image_number - 1) / w_image_number)
W = w * (w_image_number-1)+width
H = h * (h_image_number-1)+height
image = np.ones((H,W, denoise_times+2))

for t in xrange(denoise_times+2):
    for hn in xrange(h_image_number):
        for wn in xrange(w_image_number):
            #List the sub-images in rows 
            index = hn * w_image_number + wn
            if index < visualization_image_number:
                #May need transpose for w=h, but may have bugs for w ~= h
                image[hn*h:hn*h+height, wn*w:wn*w+width, t] = (output[:,index,t]).reshape((height, width))
                
sio.savemat('de-noise_visualization.mat', dict(image=image))

visualization_image_number = denoise_times+2
# layout the sub-images on a big image
w_image_number = int(math.sqrt(visualization_image_number) + 1)
h_image_number = int((visualization_image_number + w_image_number - 1) / w_image_number)
W = w * (w_image_number-1)+width
H = h * (h_image_number-1)+height
image1 = np.ones((H,W, num_show))

for n in xrange(num_show):
    for hn in xrange(h_image_number):
        for wn in xrange(w_image_number):
            #List the sub-images in rows 
            index = hn * w_image_number + wn
            if index < visualization_image_number:
                #May need transpose for w=h, but may have bugs for w ~= h
                image1[hn*h:hn*h+height, wn*w:wn*w+width, n] = (output[:,n,index]).reshape((height, width))
                
sio.savemat('de-noise_visualization1.mat', dict(image1=image1))

