# Dynamics of a Cartpole
#
# Copyright 2016 Arun Venkatraman (arunvenk@cs.cmu.edu).
#                Roberto Capobianco (capobianco@dis.uniroma1.it) 
# License provided with repository.

from math import pi, sin, cos, ceil
from functools import partial
from copy import deepcopy
import numpy as np

def rk4(t, dt, x, f):
    """ Runge-Kutta 4 implementation
    :param t:  current time
    :param dt: integration timestep
    :param x:  integration point in state space  
    :param f:  dynamics (first-order) of form xdot = f(t,x)
    :rtype tuple: (next timestep [t+dt], next point [x+dx])
    """
    k1 = f(t,x)
    k2 = f(t+0.5*dt, x+0.5*dt*k1)
    k3 = f(t+0.5*dt, x+0.5*dt*k2)  
    k4 = f(t+dt, x+dt*k3)
    x_next = x + (dt/6)*(k1 + 2.*k2 + 2.*k3 + k4);
    return t+dt, x_next 

class CartPole(object):  
    def __init__(self, x0, g=9.81, l=0.5, mp=0.1, mc=1.0, dt=0.01, ic=0.1, cd=0.2, pd=0.2):
        """
        :param x0: (position, velocity, angle, angular_velocity)
        """
        self.dt = dt
        self.params = {}
        self.params['g'] = g
        self.params['l'] = l
        
        self.params['mp'] = mp
        self.params['mc'] = mc
        self.params['ic'] = ic
        self.params['ip'] = mp*l*l
        self.params['cd'] = cd
        self.params['pd'] = pd

        self.reset(x0)

    @staticmethod
    def control_dim():
        return 1

    @staticmethod
    def state_dim():
        return 4

    def step(self, u=0, dt=0.05, sigma=np.zeros(4)):
        x = self.sensors.copy()
        self.sensors = self.transition(x, u, self.dt, self.t, self.params)
        self.t = self.t + self.dt
        x = self.sensors.copy()


        #get the last timestep and x

        if isinstance(sigma, (float, int)):
            noise = np.random.multivariate_normal(np.zeros(4), sigma*np.identity(4))
        elif len(sigma) == 4 and isinstance(sigma, np.ndarray) and (len(sigma.shape) == 1 or sigma.shape[1] == 4):
            noise = np.random.multivariate_normal(np.zeros(4), np.diag(np.array(sigma)))
        elif len(sigma) == 4 and not isinstance(sigma, np.ndarray):
            noise = np.random.multivariate_normal(np.zeros(), np.diag(np.array(sigma)))
        elif isinstance(sigma, np.ndarray) and sigma.shape == (4,4):
            noise = np.random.multivariate_normal(np.zeros(4), sigma)
        else:
            raise Exception("Process noise is either a 4x4 matrix or a 4 vector")

        x += noise

        self.states.append(x)
        self.controls.append(u)
        self.times.append(self.t)
        return x
    #end step

    def get_static_transition(self, dt):
        return partial(CartPole.transition, dt=dt, t=self.t, params=self.params) 

    @staticmethod
    def transition(x, u, dt, t, params):
        num_int_steps = 100
        dt_int = dt / float(num_int_steps)
        min_dt_int = 1e-4
        if dt_int > min_dt_int:
            num_int_steps = int(ceil(dt/min_dt_int))
            dt_int = dt / float(num_int_steps)

        dynamics = partial(CartPole._f, u = u, params=params)

        for step in range(num_int_steps):
            t,x = rk4(t, dt_int, x, dynamics)
        #endfor

        while x[2] > 8.*pi:
            x[2] -= 2.*pi

        while x[2] < -8.*pi:
            x[2] += 2.*pi


        return x
    #end transition
                        
    def reset(self, x0):
        """
        Re-initializes the environment, setting the cart back in x0.
        :param x0: (position, velocity, angle, angular_velocity)
        """
        self.t = 0.0
        self.sensors = deepcopy(x0)

        if not isinstance(self.sensors, np.ndarray):
            self.sensors = np.array(self.sensors).astype(float)
        else:
            self.sensors = self.sensors.astype(float)
        #endif

        self.states = [deepcopy(self.sensors)]
        self.controls = []
        self.times = [self.t]
    #end reset

    def get_state(self):
        """ Returns the state as (s, s', theta, theta'). """
        return self.sensors
    #end get_state

    def get_states(self):
        states = np.vstack(self.states)
        return states
    #end get_states

    def get_controls(self):
        return np.array(self.controls)
    #end get_controls

    def get_times(self):
        return np.array(self.times)
    #end get_times
        
    def get_cart_positions(self):
        """ Auxiliary access to just the cart position """
        return self.states[:, 0]
        
    def get_cart_velocities(self):
        """ Auxiliary access to just the cart velocities """
        return self.states[:, 1]

    def get_pole_angles(self):
        """ Auxiliary access to just the pole angles """
        return self.states[:, 2]

    def get_pole_velocities(self):
        """ Auxiliary access to just the pole angular velocities """
        return self.states[:, 3]

    def get_pole_cartesian_velocities(self):
        xdot = -np.cos(self.states[:, 2]) * self.states[:, 3]
        ydot = -np.cos(self.states[:, 2] - pi/2) * self.states[:, 3]

        return xdot, ydot
    #end get_pole_cartesian_velocities

    @staticmethod
    def pole_state_to_cartesian(state, l=1.0, get_x = True, get_y = True):
        """ for a single state """
        x = l*np.sin(state[2])
        y = l*np.sin(state[2] - pi/2)

        if get_x and not get_y:
            return x 
        if get_y and not get_x:
            return y 
        if get_x and get_y:
            return x, y
    #end pole_state_to_cartesian

    @staticmethod
    def cart_state_to_cartesian(state):
        """ for a single state """
        x = state[0]

        return x 
    #end pole_state_to_cartesian

    @staticmethod
    def _f(t, x, u, params): 
        """
        Dynamics in x, u at time t.
        see: http://coneural.org/florian/papers/05_cart_pole.pdf
        """
        xdot = np.zeros(4)

        sin_theta = sin(x[2])
        cos_theta = cos(x[2])
        
        mp = params['mp']
        mc = params['mc']
        l = params['l']
        ip = params['ip']
        ic = params['ic']
        cd = params['cd']
        pd = params['pd']
        g = params['g']

        xdot[0] = x[1]
        xdot[1] = (mp*l*sin_theta*x[3]*x[3] - mp*g*sin_theta*cos_theta - cd*x[1] - pd*mp*cos_theta*x[3] + u)/(mp + mc - mp*cos_theta*cos_theta)
        xdot[2] = x[3]
        xdot[3] = (-mp*l*sin_theta*cos_theta*x[3]*x[3] + (mp + mc)*g*sin_theta - cd*l*cos_theta*x[1] - pd*(mp + mc)*x[3] + cos_theta*u)/(mp*l + mc*l - mp*l*cos_theta*cos_theta)


        return xdot
    #end _f
