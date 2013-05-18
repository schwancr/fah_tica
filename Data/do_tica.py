
from msmbuilder import tICA
import sys, os, re
from msmbuilder import io
import numpy as np
from mullermsm import muller
from matplotlib.pyplot import *
import scipy
from scipy.io import mmread
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-t', dest='trajs', help='trajectory file list (file)')
parser.add_argument('-l', dest='lag', default=10, help='lag time')
parser.add_argument('-o', dest='out', default='sol.h5', help='output filename (msmbuilder.io.saveh)')
parser.add_argument('-m', dest='msm_dir', default=None, help='MSM directory to find an MSM to use to reweight the data.')

args = parser.parse_args()

traj_list = np.loadtxt(args.trajs, dtype=str)
if traj_list.shape == ():
    traj_list = np.array([traj_list], dtype=str)

if args.msm_dir is None:
    ass = None
    pops = None
else:
    ass = io.loadh(os.path.join(args.msm_dir, 'Assignments.Fixed.h5'), 'arr_0')
    pops = np.loadtxt(os.path.join(args.msm_dir, 'Populations.dat'))
    tProb = mmread(os.path.join(args.msm_dir, 'tProb.mtx'))

c = tICA.CovarianceMatrix(args.lag, tProb=tProb, populations=pops)

for i, fn in enumerate(traj_list):
    print fn
    t = np.load(fn)
    c.train(t, ass[i])
    #c.train(t)

C, Sigma = c.get_current_estimate()

vals, vecs = scipy.linalg.eig(C, b=Sigma)

print vals
print vecs

io.saveh(args.out, vals=vals, vecs=vecs, C=C, Sigma=Sigma)

muller.plot_v()


ref = io.loadh('ref.h5')
ref['vecs'][:,0] *= -1

vecs[:,0] *= -1

plot([0, vecs[0,0]], [0, vecs[1,0]], color='white', lw=3)
plot([0, vecs[0,1]], [0, vecs[1,1]], color='white', ls='dashed', lw=3)

plot([0, ref['vecs'][0,0]], [0, ref['vecs'][1,0]], color='red', lw=3)
plot([0, ref['vecs'][0,1]], [0, ref['vecs'][1,1]], color='red', ls='dashed', lw=3)

show()
