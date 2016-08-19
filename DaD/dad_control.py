#
# Data as Demonstrator (DaD) + Control
#
# Implements the ISER 2016 paper: (http://www.cs.cmu.edu/~arunvenk/papers/2016/Venkatraman_iser_16.pdf)
#   DaD originally presented at AAAI 2015.  (https://www.ri.cmu.edu/pub_files/2015/1/Venkatraman.pdf)
#
# Copyright 2016 Arun Venkatraman (arunvenk@cs.cmu.edu).
#                Roberto Capobianco (capobianco@dis.uniroma1.it) 
# License provided with repository.
#

from copy import deepcopy
import numpy as np
import helpers.pyhelpers as pyhelpers

from IPython import embed # noqa

__author__ = 'arunvenk'

class DaDControl(object):
    """Data as Demonstrator + Control class.  """

    def __init__(self, traj_pwr_filter = 1.0, rollout_err_filter = 1.5, max_train_size = 200000, max_data_size = 900000):
        """
        Rolling out the learned model can produce data points with very large error. To make the learning problem
        easier, we filter out points using a static threshold (traj_pwr_filter) which scales the RMS power of the
        training trajectories and a dynamics threshold (rollout_err_filter) which scales the RMS error on the rollout
        from this iteration.

        To make the learning computationally tractable, we subsample uniformly max_train_size points to learn with every 
        iteration of DaD. To prevent memory issues, we keep only max_data_size points at the end of every iteration.

        :param traj_pwr_filter: Scaling on mean_traj_pwr for selecting new data points to keep.
        :param rollout_err_filter: Scaling on mean_train_rollout_err for selecting new data points to keep.
        :param max_train_size: Number of data points to learn with in each iteration. Data is subsampled for learning if 
            aggregated dataset gets larger than this size.
        :param max_data_size: Maximum number of data points to keep in the aggregated dataset. The aggregated dataset 
            is subsampled to this size if it exceeds it.
        """
        # 
        self.TRAJ_PWR_SCALE = traj_pwr_filter
        self.TRAIN_ROLLOUT_ERR_SCALE = rollout_err_filter
        # The maximum number of aggregated data points to keep. If exceed, random 
        # permutation of data points is kept between DAgger iterations
        self.MAX_TRAIN_SIZE = max_train_size;
        self.MAX_DATA_SIZE = max_data_size

    def learn(self, Xs, Us, learner, dagger_iters = 20, Xtest = None, Utest = None, verbose = True):
        """ Trains a Dynamical System using Data As Demonstrator + Control 

        Uses "learner" to fit a dynamics model to predict x_{t+1} from (x_t, u_t). We provide helper wrapper classes in
        helpers.learner_wrapper to assist in using learning algorithms.

        :param Xs - States of dimension [timesteps x dim_x x num_traj]
        :param Us - Controls of dimension [timesteps-1 x dim_u x num_traj]
        :param learner - object with functions .fit(X_t,U_t,X_{t+1}) and .predict(X_t, U_t)
            where X_t, X_{t+1} are of dim [num_pts, dim_x] and U_t is of [num_pts, dim_u].
        :param dagger_iters - number of DAgger iterations for optimizing the 
        :param Xtest [optional] - States of dimension [timesteps x dim_y x num_traj]
        :param Utest [optional] - Controls of dimension [timesteps-1 x dim_u x num_traj]
            Required if Xtest is passed in
        """
        verboseprint = pyhelpers.get_verboseprint(verbose)
        # store reference to learner 
        self.learner = learner
        
        T = Xs.shape[0]
        dim_x = Xs.shape[1]
        num_traj = Xs.shape[2]
        dim_u = Us.shape[1]

        Xt = pyhelpers.tensor_to_dataset(Xs[:-1,:,:]) #[num_traj*timesteps, dim_data]
        Xt1 = pyhelpers.tensor_to_dataset(Xs[1:,:,:])
        Ut = pyhelpers.tensor_to_dataset(Us)

        self.mean_traj_pwr = np.mean(pyhelpers.rms_error(Xs, np.zeros(Xs.shape)))
        Xt1_gt = Xt1.copy() # these will be ground truth targets
        X0s = Xs[0,:,:] # [dim_x, num_traj]

        has_test_data = False
        if Xtest is not None:
            has_test_data = True
            X0s_test = Xtest[0,:,:]
            num_test_traj = Xtest.shape[2]
            self.mean_test_traj_pwr = np.mean(pyhelpers.rms_error(Xtest, np.zeros(Xtest.shape)))

        verboseprint('Training initial model...')
        learner.fit(Xt, Ut, Xt1)
        self.initial_model = deepcopy(learner) 
        # Find the initial training error.
        _, errors = self.test(Xs, Us, learner)
        self.mean_train_rollout_err = np.mean(errors)
        self.initial_train_error = self.mean_train_rollout_err 
        self.min_train_error = self.mean_train_rollout_err
        self.min_train_error_model = deepcopy(learner)

        # Find the initial test error.
        self.min_test_error = np.Inf; self.min_test_error_model = deepcopy(learner)
        if has_test_data:
            _, errors = self.test(Xtest, Utest, learner)
            self.initial_test_err = np.mean(errors)
            self.min_test_error = self.initial_test_err

        # Start the DaD Main loop
        train_errors = []; test_errors = []
        for i in xrange(1, dagger_iters+1): #iteration
            verboseprint(">DaD Iteration: {}/{}".format(i, dagger_iters))
            # Get the predictions [timesteps, dim_x, num_traj].
            Xpred = self._rollout_model(X0s, Us)
            self.mean_train_rollout_err = np.mean(pyhelpers.rms_error(Xpred[:-1,:,:], Xs[1:,:,:]))
            if self.mean_train_rollout_err < self.min_train_error:
                self.min_train_error = self.mean_train_rollout_err 
                self.min_train_error_model = deepcopy(learner) 
            train_errors.append(self.mean_train_rollout_err)
            verboseprint('  Training error: {0:.4g}. Train Pwr: {1:.4g}'.format(self.mean_train_rollout_err, self.mean_traj_pwr))
            if has_test_data:
                Xpred_test = self._rollout_model(X0s_test, Utest)
                mean_test_rollout_err = np.mean(pyhelpers.rms_error(Xpred_test[:-1,:,:], Xtest[1:,:,:]))
                test_errors.append(mean_test_rollout_err)
                if mean_test_rollout_err < self.min_test_error:
                    self.min_test_error = mean_test_rollout_err 
                    self.min_test_error_model = deepcopy(learner) 
                verboseprint('  Testing error: {0:.4g}. Test Pwr: {1:.4g}. Min test error:{2:.4g}'.format(mean_test_rollout_err,
                            self.mean_test_traj_pwr, self.min_test_error))

            # We don't have ground truth for the t+1 prediction so do to -1.
            xt_hat = pyhelpers.tensor_to_dataset(Xpred[:-1,:,:]) 
            keep_inds = self._remove_large_err_samples(Xt1_gt, xt_hat)
            num_total = Xt1_gt.shape[0]
            num_kept = keep_inds.size 
            verboseprint(" Keeping {:d}/{:d}={:3.2f}% of the data".format(num_kept, num_total, 
                float(num_kept) / float(num_total) * 100))
            Xt = np.concatenate((Xt, xt_hat[keep_inds]))
            Xt1 = np.concatenate((Xt1, Xt1_gt[keep_inds]))
            Ut = np.concatenate((Ut, Ut[keep_inds]))
            # if we exceed the max size we can train on
            train_xt = Xt; train_ut = Ut; train_xt1 = Xt1;
            if Xt.shape[0] > self.MAX_TRAIN_SIZE:
                perm = np.random.choice(Xt.shape[0], self.MAX_TRAIN_SIZE, replace = False);
                train_xt = Xt[perm]; train_xt1 = Xt1[perm]; train_ut = Ut[perm]
            learner.fit(train_xt, train_ut, train_xt1)

            # If we exceed the max memory for holding all data, then subsample it.
            if Xt.shape[0] > self.MAX_DATA_SIZE:
                perm = np.random.choice(Xt.shape[0], self.MAX_DATA_SIZE, replace = False);
                Xt = Xt[perm]; Xt1 = Xt1[perm]; Ut = Ut[perm]
            verboseprint(' Dataset Size: {:d}.'.format(Xt.shape[0]))
        #end for

        return train_errors, test_errors


    def test(self, Xtest, Utest, learner=None):
        """ Test rollouts along from the first state of Xtest following the control sequence in Utest. 
        
        Uses the best learned model to generate rollouts from Xtest[0,:,:] (the first state in each trajectory) while
        using the predetermined control sequence Utest. Returns the predictions and RMS error between the predicted 
        states and the true states Xtest.

        :param Xtest - States of dimension [timesteps x dim_x x num_traj]
        :param Utest - Controls of dimension [timesteps x dim_u x num_traj]
        :param learner - Learner to use for the test.
        """
        if learner is None:
            learner = self.min_test_error_model
        X0s_test = Xtest[0,:,:]
        Xpred_test = self._rollout_model(X0s_test, Utest, learner)
        test_rollout_err = pyhelpers.rms_error(Xpred_test[:-1,:,:], Xtest[1:,:,:])
        return Xpred_test, test_rollout_err

    def _rollout_model(self, X0s, Us, learner=None):
        """ Rollout the learned predictor from initial states, X0s, with controls Us 

        :param X0s - States, [dim_x, num_traj]
        :param Us - controls, [timesteps, dim_u, num_traj]
        :rtype - predictions [timesteps, dim_x, num_traj]
        """
        T = Us.shape[0]
        num_traj = Us.shape[2]
        dim_x = X0s.shape[0]
        if learner is None:
            learner = self.learner
        # prevent '.' lookups for speed
        predict = learner.predict
        predictions = np.zeros((T+1, dim_x, num_traj)) + np.NaN
        # initialize the current X_t
        Xt = X0s.T # make it [num_traj, dim_x]
        predictions[0,:,:] = Xt.T
        for t in xrange(0, T):
            Ut = Us[t, :, :].T
            Xt1 = predict(Xt, Ut) # should be [num_traj, dim_x]
            predictions[t+1,:,:] = Xt1.T 
            Xt = Xt1
        return predictions

    def _remove_large_err_samples(self, Xtgt, Xthat): 
        """ Removes data points from being added to the aggregated dataset if they were in too much error.  
        
        :param Xtgt - [num_data_pts, dim_x]
        :param Xthat - [num_data_pts, dim_x]
        :rtype np.array - indices to keep
        """
        # compute the error
        error = Xtgt - Xthat 
        l2_err = np.sqrt(np.sum(error * error, 1))
        keep_inds = np.where(np.logical_and(l2_err < self.TRAJ_PWR_SCALE*self.mean_traj_pwr,
            l2_err < self.TRAIN_ROLLOUT_ERR_SCALE*self.mean_train_rollout_err))[0]
        return keep_inds



