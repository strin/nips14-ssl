import gpulearn_mm_z_x_stacked
import sys
import time

if 'svhn' in sys.argv[1]:
    gpulearn_mm_z_x_stacked.main(dataset=sys.argv[1], n_z=300, n_hidden=(500,500), seed=0, comment='', gfx=True)
elif sys.argv[1] == 'mnist':
    stack_number = 1
    nn_hidden = tuple(stack_number *[(500,)])
    n_z = tuple(stack_number * [200])
    c = tuple(stack_number * [0])
    str_t = str(nn_hidden)+'_'+str(int(time.time()))
    for i in xrange(stack_number):
        gpulearn_mm_z_x_stacked.main(dataset='mnist', n_z=n_z[i], c=c[i], nn_hidden=nn_hidden, seed=0, comment='', gfx=True, encoder_index = i, str_t = str_t)

#gpulearn_z_x.main(n_data=50000, dataset='svhn_pca', n_z=300, n_hidden=(500,500), seed=0)
