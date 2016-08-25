#!/usr/bin/env python

# Demo for learning the dynamics of a cartpole under a linear policy
#
# Copyright 2016 Arun Venkatraman (arunvenk@cs.cmu.edu).
#                Roberto Capobianco (capobianco@dis.uniroma1.it) 
# License provided with repository.

from cartpole import CartPole
import os, sys
import numpy as np

DAD_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(DAD_MODULE_PATH)

from DaD.dad_control import DaDControl
from DaD.helpers.learner_wrapper import DynamicsControlDeltaWrapper
from sklearn.linear_model.ridge import Ridge

# Choose the system to use. Needs to follow a common api.
SYSTEM = CartPole

def x0_low_high(system):
    """Lower and higher bound for random intitial state generation. """
    if system == CartPole:
        x0_low = np.array([-1, -1, -np.pi, -0.5])
        x0_high = np.array([1, 1, np.pi, 0.5])
    else:
        raise Exception('No x0 bounds defined for this sytem') 
    return x0_low, x0_high

def run_episodes(policy, num_episodes, T):
    """Generate num_episodes runs of the SYSTEM. """
    # Lower and higher bound for random intitial state generation.
    x0_low, x0_high = x0_low_high(SYSTEM)
    all_states = []; all_actions = []; 
    for i in range(num_episodes):
        x0 = x0_low + (x0_high-x0_low)*np.random.random(4)
        states, actions = run_trial(policy, T, x0)
        all_states.append(states)
        all_actions.append(actions)
    return np.stack(all_states, axis=2), np.dstack(all_actions).transpose((1,0,2))

def run_trial(policy, T, x0 = np.array((0, 0, np.pi/2., 0))):
    """Generate T timesteps of data from the SYSTEM. """
    # Initialize the system at x0.
    system = SYSTEM(x0)
    DT = 0.10 # simulate at 10 Hz
    xt = x0.copy()
    for t in xrange(T):
        ut = policy.u(xt) 
        xt = system.step(DT, ut, sigma=1e-3+np.zeros(SYSTEM.state_dim()))
    X = system.get_states()
    U = system.get_controls()
    return X, U

class RandomLinearPolicy(object):
    """Generates a N-dimensional random control within a range. """
    def __init__(self, state_dim, control_dim, u_min = -1.0, u_max = 1.0):
        self.A = np.random.random((control_dim, state_dim))
    def u(self, state):
        #return self.u_min + (self.u_max-self.u_min)*np.random.rand(self.control_dim)
        return np.dot(self.A, state)
    

def optimize_learner_dad(learner, X, U, iters, train_size = 0.5):
    num_traj = X.shape[2]
    if train_size < 1.0:
        from sklearn import cross_validation
        rs = cross_validation.ShuffleSplit(num_traj, n_iter=1, train_size=train_size, 
                random_state=0, test_size=1.-train_size)
        for train_index, test_index in rs:
            pass
        Xtrain = X[:,:,train_index]; Xtest = X[:,:,test_index]
        Utrain = U[:,:,train_index]; Utest = U[:,:,test_index]
    elif train_size == 1.0:
        Xtrain = X; Xtest = X
        Utrain = U; Utest = U
    else:
        raise Exception('Train size must be in (0,1]')

    dad = DaDControl()
    dad.learn(Xtrain, Utrain, learner, iters, Xtest, Utest, verbose=False)
    print(' DaD (iters:{:d}). Initial Err: {:.4g}, Best: {:.4g}'.format(iters,
        dad.initial_test_err, dad.min_test_error))
    return dad


if __name__ == "__main__":
    print('Defining the learner')
    learner = DynamicsControlDeltaWrapper(Ridge(alpha=1e-4, fit_intercept=True))

    NUM_EPISODES = 50
    T = 50
    print('Generating train data')
    policy = RandomLinearPolicy(SYSTEM.state_dim(), SYSTEM.control_dim())

    Xtrain, Utrain = run_episodes(policy, NUM_EPISODES, T)
    print('Generating test data')
    Xtest, Utest = run_episodes(policy, NUM_EPISODES, T)

    print('\nLearning dynamics')
    iters = 25
    dad = optimize_learner_dad(learner, Xtrain, Utrain, iters, train_size=0.5)

    _, dad_err = dad.test(Xtest, Utest, dad.min_test_error_model)
    _, non_dad_err = dad.test(Xtest, Utest, dad.initial_model)
    print(' Err without DaD: {:.4g}, Err with DaD: {:.4g}'.format(np.mean(non_dad_err), np.mean(dad_err)))

