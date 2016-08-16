#
# Learner Wrapper helper classes for Data as Demonstrator (DaD) repository.
#
# Copyright 2016 Arun Venkatraman (arunvenk@cs.cmu.edu).
# License provided with repository.
#

import numpy as np
import pyhelpers as pyh

class DynamicsControlWrapper(object):
    """Wrapper class to use a learner (e.g. from sklearn) with DaD+Control that learns a transition model f(x_t, u_t) -> x_{t+1}. 

    Takes call to .fit(X_t,U_t,X_{t+1}) and calls learner.fit(X,Y) 
    Takes call to .predict(X_t, U_t) and calls learner.predict(X) 
    """
    def __init__(self, learner):
        self.learner = learner

    def fit(self, Xt, Ut, Xt1):
        """
        :param Xt - [num_data_pts, dim_x] matrix
        :param Ut - [num_data_pts, dim_u] matrix
        :param Xt1 - [num_data_pts, dim_x] matrix
        """
        Xt = pyh.ensure_2d(Xt)
        Ut = pyh.ensure_2d(Ut)
        inputs = np.hstack((Xt, Ut))
        self.learner.fit(inputs, Xt1)

    def predict(self, Xt, Ut):
        if len(Xt.shape) == 1:
            inputs = np.expand_dims(np.hstack((Xt, Ut)), axis=0)
            Xt1 = self.learner.predict(inputs)
            return Xt1.ravel()
        inputs = np.hstack((Xt, Ut))
        Xt1 = self.learner.predict(inputs)
        return Xt1

class DynamicsControlDeltaWrapper(object):
    """Wrapper class to use a learner (e.g. from sklearn) with DaD+Control that works to predict state deltas.

    Specifically, learns a transition model f(x_t, u_t) -> \Delta_x, where \Delta_x = x_{t+1} - x_t .
    This centers the mean for the learner's predictions around the previous prediction (x_t) instead around 0, similar
    to the approach taken in PILCO. This can make the learning problem easier.
    
    Takes call to .fit(X_t,U_t,X_{t+1}) and calls learner.fit(X,Y) 
    Takes call to .predict(X_t, U_t) and calls learner.predict(X) 
    """
    def __init__(self, learner):
        self.learner = learner

    def fit(self, Xt, Ut, Xt1):
        """
        :param Xt - [num_data_pts, dim_x] matrix
        :param Ut - [num_data_pts, dim_u] matrix
        :param Xt1 - [num_data_pts, dim_x] matrix
        """
        Xt = pyh.ensure_2d(Xt)
        Xt1 = pyh.ensure_2d(Xt1)
        Ut = pyh.ensure_2d(Ut)
        inputs = np.hstack((Xt, Ut))
        dX = Xt1 - Xt
        self.learner.fit(inputs, dX)

    def predict(self, Xt, Ut):
        if len(Xt.shape) == 1:
            inputs = np.expand_dims(np.hstack((Xt, Ut)), axis=0)
            dX = self.learner.predict(inputs)
            Xt1 = Xt + dX
            return Xt1.ravel()
        inputs = np.hstack((Xt, Ut))
        dX = self.learner.predict(inputs)
        Xt1 = Xt + dX
        return Xt1

