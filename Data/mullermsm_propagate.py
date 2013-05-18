#!/usr/bin/env python
import os, sys
import argparse
import mullermsm
from mullermsm import metric
from mullermsm import muller
from msmbuilder import Project, Trajectory
import numpy as np
import IPython as ip
import random
import cPickle as pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_trajs', help='number of trajectories. Default=10', type=int, default=10)
    parser.add_argument('-t', '--traj_length', help='trajectories length. Default=10000', type=int, default=10000)
    args = parser.parse_args()
    
    # these could be configured
    kT = 15.0
    dt = 0.1
    mGamma = 1000.0
    
    forcecalculator = muller.muller_force()
    

    #project = Project({'conf_filename': os.path.join(mullermsm.__path__[0], 'conf.pdb'),
    #          'num_trajs': args.n_trajs,
    #          'project_root_dir': '.',
    #          'traj_file_base_name': 'trj',
    #          'traj_file_path': 'Trajectories',
    #          'traj_file_type': '.lh5',
    #          'traj_lengths': [args.traj_length]*args.n_trajs})
     
    #if os.path.exists('ProjectInfo.h5'):
    #    print >> sys.stderr, "The file ./ProjectInfo.h5 already exists. I don't want to overwrite anything, so i'm backing off"
    #    sys.exit(1)
    
    
    try:
        os.mkdir('Trajectories')
    except OSError:
        print >> sys.stderr, "The directory ./Trajectores already exists. I don't want to overwrite anything, so i'm backing off"
        sys.exit(1)
        
    for i in range(args.n_trajs):
        print 'simulating traj %s' % i
        
        # select initial configs randomly from a 2D box
        initial_x = [0,0]
        initial_x = [random.uniform(-1.5, 1.2), random.uniform(-0.2, 2)]
        print 'starting conformation from randomly sampled points (%s, %s)' % (initial_x[0], initial_x[1])
        print 'propagating for %s steps on the Muller potential with a Langevin integrator...' % args.traj_length
        
        positions = muller.propagate(args.traj_length, initial_x, kT, dt, mGamma, forcecalculator)

        # positions is N x 2, but we want to make it N x 1 x 3 where the additional
        # column is just zeros. This way, being N x 1 x 3, it looks like a regular MD
        # trajectory that would be N_frames x N_atoms x 3
        positions3 = np.hstack((positions, np.zeros((len(positions),1)))).reshape((len(positions), 1, 3))
        t = Trajectory.LoadTrajectoryFile(project['ConfFilename'])
        t['XYZList'] = positions3
        
        t.SaveToLHDF(project.GetTrajFilename(i))
        print 'saving trajectory to %s' % project.GetTrajFilename(i)
        
    project.SaveToHDF('ProjectInfo.h5')
    print 'saved ProjectInfo.h5 file'

    
    pickle.dump(metric.EuclideanMetric(), open('metric.pickl', 'w'))
    print 'saved metric.pickl'
    
    
if __name__ == '__main__':
    main()
