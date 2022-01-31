"""
Double Integrator with noise in observations.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import scipy.stats as stats
import sympy as sp
import numpy as np
from sympy.physics.vector import dynamicsymbols as dynamicsymbols
import IPython as ipy
from filterpy.kalman import KalmanFilter


class DoubleIntegratorEnv(gym.Env):
    """
    Description:
        Double integrator

    Observation:
        Type: Box(2)

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right


    Reward:
        
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, noise_scale=0.001): 

        # self.kinematics_integrator = 'euler'
        self.kinematics_integrator = 'semi-implicit'

        self.nx = 2 # Number of states
        self.ny = self.nx # Number of observations
        self.nu = 3 # Number of control inputs
        self.force_mag = 10.0 # scaling for control input

        self.tau = 0.1 # Time step
        self.T = 5 # 5 # 10 # Time horizon

        self.action_space = spaces.Discrete(self.nu)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.ny), np.inf*np.ones(self.ny), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.t = None

        self.x_threshold = 1.0
        self.x_dot_threshold = 1.0
        self.x_range = [-self.x_threshold, self.x_threshold]
        self.x_dot_range = [-self.x_dot_threshold, self.x_dot_threshold]

        # Std. dev. of observation noise (pos, vel)
        self.noise_scale = noise_scale 
        self.noise_std_dev = self.noise_scale*np.array([1.0, 1.0])


        # Setup Kalman filter
        self.kalman_filter = True
        self.x0_belief_std_dev = 1.0*np.array([self.x_threshold, self.x_dot_threshold]) 
        if self.kalman_filter:
            # A and B matrices for linear system
            if self.kinematics_integrator == 'euler':
                A = np.array([[1,self.tau],[0,1]])
                B = np.array([[0,self.tau]]).T
            elif self.kinematics_integrator == 'semi-implicit':
                A = np.array([[1,self.tau],[0,1]])
                B = np.array([[self.tau**2,self.tau]]).T
            else:
                raise Exception("Integrator not recognized.")

            filter = KalmanFilter(dim_x=self.nx, dim_z=self.ny)
            filter.x = np.zeros((self.nx,1)) # Initial state estimate
            filter.P = np.diag(self.x0_belief_std_dev**2) # covariance of initial belief
            filter.Q = 0.0*np.eye(self.nx) # Process noise
            filter.R = np.diag(self.noise_std_dev**2) # Measurement noise
            filter.H = np.eye(self.nx) # Measurement function
            filter.F = A # State transition matrix
            filter.B = B # Control matrix
            self.filter = filter


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_noise(self, obs):
        noise = np.random.normal(np.zeros_like(obs), self.noise_std_dev)
        obs_w_noise = obs + noise
        return obs_w_noise

    def get_p_y_x(self, observations, states): 
        if (len(states.shape) == 1): # Single query
            observations = np.reshape(observations, (self.ny, 1))
            states = np.reshape(states, (self.nx, 1))

        # Vectorized computation of p_y_x. Expects arrays of shape (nx, num_samples).
        num_samples = states.shape[1]
        noises = np.repeat(np.reshape(self.noise_std_dev,(self.nx,1)), num_samples, 1) 
        p_ys_xs = np.prod(stats.norm.pdf(observations, states, noises),0)

        return p_ys_xs


    def step(self, action):

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot = self.state
        # u = self.force_mag if action == 1 else -self.force_mag
        if action == 0:
            u = 0.0
        elif action == 1:
            u = self.force_mag
        else: #  action == 2:
            u = -self.force_mag
        # elif action == 3:
        #     u = -0.5*self.force_mag
        # else:
        #     u = -self.force_mag

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * u
        elif self.kinematics_integrator == 'semi-implicit': # semi-implicit euler
            x_dot = x_dot + self.tau * u
            x = x + self.tau * x_dot
        else:
            raise Exception("Integrator not recognized.")

        self.state = (x, x_dot)

        # out_of_bounds = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        #     or theta < -self.theta_threshold_radians
        #     or theta > self.theta_threshold_radians
        # )

        # Check if we have gone beyond time horizon
        if (self.t > (self.T-1)):
            self.steps_beyond_done = self.t - (self.T-1)
            done = True
        else:
            done = False

        if done: # done only if beyond time horizon
            reward = 0.0
        else:
            # reward = 1 - out_of_bounds

            reward_x = min(-x+self.x_threshold, x+self.x_threshold)
            reward_x = reward_x/self.x_threshold
            reward_x = min(reward_x, 0.8)/0.8
            reward_x = max(0.0, reward_x)

            reward_x_dot = min(-x_dot+self.x_dot_threshold, x_dot+self.x_dot_threshold)
            reward_x_dot = reward_x_dot/self.x_dot_threshold
            reward_x_dot = min(reward_x_dot, 0.8)/0.8
            reward_x_dot = max(0.0, reward_x_dot)

            reward = (reward_x + reward_x_dot)/2

            if reward > 1:
                ipy.embed()

        obs_with_noise = self.add_noise(np.array(self.state))

        # Kalman filter
        if self.kalman_filter:
            self.filter.predict(u=u)
            self.filter.update(obs_with_noise)
            state_estimate = np.reshape(self.filter.x, (self.nx,))
            obs_with_noise = state_estimate

        # Update time
        self.t += 1

        return obs_with_noise, reward, done, {}

    def reset(self):
        self.t = 0

        # Uniform distribution
        self.state = self.np_random.uniform(low=[self.x_range[0], self.x_dot_range[0]], high=[self.x_range[1],self.x_dot_range[1]])

        # # Gaussian distribution
        # self.state = self.np_random.normal(np.zeros(self.nx), self.x0_belief_std_dev)

        # Generate observation
        self.steps_beyond_done = None
        obs_w_noise = self.add_noise(np.array(self.state))

        # Reset filter
        if self.kalman_filter:
            self.filter.x = np.zeros((self.nx,1)) # Initial state estimate
            self.filter.P = np.diag(self.x0_belief_std_dev**2) # covariance of initial belief

            # Update filter using observation
            self.filter.update(obs_w_noise)
            state_estimate = np.reshape(self.filter.x, (self.nx,))
            obs_w_noise = state_estimate

        
        return obs_w_noise

    def reset_state(self, state):
        self.t = 0
        # Reset at particular state
        self.state = state

        done = False
        self.steps_beyond_done = None

        obs_w_noise = self.add_noise(np.array(self.state))
        return obs_w_noise, done

    def is_done(self, state):
        done = False # there are no done states here (unless t is more than horizon)
        return done


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART

        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            pole.add_attr(self.carttrans)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
