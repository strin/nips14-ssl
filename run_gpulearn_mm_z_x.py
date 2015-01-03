import gpulearn_mm_z_x
import sys, os

if 'svhn' in sys.argv[1]:
    gpulearn_mm_z_x.main(dataset=sys.argv[1], n_z=300, n_hidden=(500,500), seed=0, comment='', gfx=True)
elif sys.argv[1] == 'mnist':
    n_hidden = (500,500)
    if len(sys.argv) > 2:
      n_hidden = tuple([int(x) for x in sys.argv[2:]])
    nz=500
    if os.environ.has_key('nz'):
      nz = int(os.environ['nz'])
    gpulearn_mm_z_x.main(dataset='mnist', n_z=nz, n_hidden=n_hidden, seed=0, comment='', gfx=True)

#gpulearn_z_x.main(n_data=50000, dataset='svhn_pca', n_z=300, n_hidden=(500,500), seed=0)
