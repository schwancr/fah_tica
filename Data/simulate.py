import random
from mullermsm import muller
mullerforce = muller.muller_force()
import scipy.linalg
from matplotlib.pyplot import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dt', dest='dt', type=float, default=0.1)
parser.add_argument('-n', dest='num_frames', type=int, default=100000)
parser.add_argument('-o', dest='output', default='pos.npy')
parser.add_argument('-i', dest='init_x', default=None, nargs='+', help='initial positions')
args = parser.parse_args()

kT = 15.0
dt = args.dt
mGamma = 1000.0
traj_length = args.num_frames 
if args.init_x is None:
    initial_x = [random.uniform(-1.5, 1.2), random.uniform(-0.2, 2)]
else:
    initial_x = [np.float(args.init_x[0]), np.float(args.init_x[1])]

positions = muller.propagate(traj_length, initial_x, kT, dt, mGamma, mullerforce)

np.save(args.output, positions)

#muller.plot_v()
#plot(positions[:,0], positions[:,1], color='white', lw=3)
#show()
