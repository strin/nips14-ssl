import gpulearn_yz_x
import sys, os

n_hidden = (500,500)
if len(sys.argv) > 2:
  n_hidden = tuple([int(x) for x in sys.argv[2:]])
nz=500
if os.environ.has_key('nz'):
  nz = int(os.environ['nz'])
gpulearn_yz_x.main(dataset=sys.argv[1], n_z=nz, n_hidden=n_hidden, seed=0, gfx=True)
