import gpulearn_mm_z_x
import sys, os

n_hidden = (500,500)
if len(sys.argv) > 2:
  n_hidden = tuple([int(x) for x in sys.argv[2:]])
nz=500
if os.environ.has_key('nz'):
  nz = int(os.environ['nz'])
if os.environ.has_key('stepsize'):
  alpha = float(os.environ['stepsize'])
else:
  alpha = 3e-4
if os.environ.has_key('decay1'):
  decay1 = float(os.environ['decay1'])
else:
  decay1 = 0.1
if os.environ.has_key('decay2'):
  decay2 = float(os.environ['decay2'])
else:
  decay2 = 0.001
  
gpulearn_mm_z_x.main(dataset=sys.argv[1], n_z=nz, n_hidden=n_hidden, seed=0, comment='', alpha=alpha, decay1=decay1, decay2=decay2, gfx=True)


#gpulearn_z_x.main(n_data=50000, dataset='svhn_pca', n_z=300, n_hidden=(500,500), seed=0)
