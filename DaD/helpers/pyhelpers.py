#
# Helper functions for Data as Demonstrator (DaD) repository.
#
# Copyright 2016 Arun Venkatraman (arunvenk@cs.cmu.edu).
# License provided with repository.
#

import numpy as np
import os
import scipy.io as sio

from IPython import embed # noqa

def load_dataset(data_fname):
    if not os.path.isfile(data_fname):
        raise IOError('Cannot find file: {}'.format(data_fname))
    if os.path.splitext(data_fname)[1] == '.mat':
        data = sio.loadmat(data_fname)
    elif os.path.splitext(data_fname)[1] == '.npz':
        data = np.load(data_fname)
    else:
        raise Exception('Unknown input data extension')
    return data

def tensor_to_dataset(traj_tensor):
    """
    :param traj_tensor : Tensor representing trajectory [timesteps, dim_data, num_traj]
    :rtype : Matrix of [num_traj*timesteps, dim_data]
    """
    mat = np.vstack(traj_tensor.transpose((0,2,1)))
    return mat

def rms_error(trajs_a, trajs_b):
    """Computes the RMS L2 error between trajectories.

    :param trajs_a: Matrix representing trajectory [timesteps, dim_data]
                or tensor [timesteps, dim_data, num_traj]
    :param trajs_b: Matrix representing trajectory [timesteps, dim_data]
                or tensor [timesteps, dim_data, num_traj]. 
            Currently Note: trajs_b must be same shape as traj1
    :rtype double or numpy array
    :return RMS error per trajectory
    """
    def _rms_traj(traj1, traj2):
        """Helper function to compute the rms error between two trajectories. """
        err = (traj1- traj2)
        sq_err = err*err
        rms_err = np.sqrt(np.mean(np.sum(sq_err, axis=1)))
        return rms_err
    # If we havve more than one trajectory, compute the error across all the trajectories.
    if len(trajs_a.shape) == 3: 
        rms = np.array([_rms_traj(trajs_a[:,:,n], trajs_b[:,:,n]) for n in xrange(trajs_a.shape[2])])
    else:
        rms = _rms_traj(trajs_a, trajs_b) 
    return rms

def ensure_2d(X, axis=1):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=axis)
    elif len(X.shape) == 0:
        X = np.expand_dims(np.expand_dims(X, axis=axis), axis=axis)
    return X

def get_verboseprint(verbose):
    """This function is from: http://stackoverflow.com/a/5980173 """
    if verbose:
        def verboseprint(*args):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
               print arg,
            print
    else:   
        verboseprint = lambda *a: None      # do-nothing function
    return verboseprint
