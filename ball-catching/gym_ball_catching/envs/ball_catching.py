"""
Ball Catching with noise in observations.
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


class BallCatchingEnv(gym.Env):
    """
    Description:
        Robot catching a ball using a noisy sensor.
        
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, noise_scale=1.0): 

        self.kinematics_integrator = 'euler'

        self.nx = 4 # Number of states
        self.ny = self.nx # Number of observations

        # Control inputs
        self.inputs = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        self.nu = len(self.inputs) # Number of control inputs

        self.tau = 1.0 # Time step
        self.T = 4 # Time horizon;  This is (T-1) using the paper's formulation.
        self.g = 0.1 # Chosen so that ball hits ground around T = 5

        # Reward threshold
        self.reward_thresh = 0.5 # If relative position is more than this, we get 0 reward (if relative position is 0, we get reward = 1)

        self.action_space = spaces.Discrete(self.nu)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.ny), np.inf*np.ones(self.ny), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.t = None

        # Range of initial conditions for relative positions and ball velocity
        self.x_rel_range = [-0.01, 0.01]
        self.vx_ball_range = [-0.2, 0.2]
        self.y_rel_range = [1.0, 1.1]
        self.vy_ball_range = [0.0, 0.1] 
        self.initial_state_estimate = np.array([[np.mean(self.x_rel_range), np.mean(self.vx_ball_range), np.mean(self.y_rel_range), np.mean(self.vy_ball_range)]]).T

        self.x0_mu = np.array([np.mean(self.x_rel_range), np.mean(self.vx_ball_range), np.mean(self.y_rel_range), np.mean(self.vy_ball_range)])
        self.x0_belief_std_dev = 1.0*np.array([self.x_rel_range[1], self.vx_ball_range[1], self.y_rel_range[1]-self.y_rel_range[0], self.vy_ball_range[1]]) 
        self.x0_cov = np.diag(self.x0_belief_std_dev**2)

        # Std. dev. of observation noise
        self.noise_scale = noise_scale 
        self.noise_std_dev = self.noise_scale*np.array([0.5, 0.75, 1.0, 1.0])

        # Setup Kalman filter
        self.kalman_filter = True

        # Expected reward at time 0
        self.R0_expected = self.compute_expected_reward([self.x0_mu], self.x0_cov)

        if self.kalman_filter:
            # A and B matrices for linear system
            if self.kinematics_integrator == 'euler':
                A = np.array([[1,self.tau, 0, 0],[0, 1, 0, 0],[0, 0, 1, self.tau],[0, 0, 0, 1]])
                self.A_transition = A
                # Treat gravity as control input (to write dynamics as linear instead of affine)
                B = np.array([[-self.tau,0, 0, 0],[0,0,0,-self.tau]]).T 
            else:
                raise Exception("Integrator not recognized.")

            filter = KalmanFilter(dim_x=self.nx, dim_z=self.ny)
            filter.x = self.initial_state_estimate # Initial state estimate
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
        # This function is only used for RL (not computing the bound; that is done analytically)

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x_rel, vx_ball, y_rel, vy_ball = self.state
        u = self.inputs[action]

        if self.kinematics_integrator == 'euler':
            x_rel = x_rel + self.tau * vx_ball - self.tau*u
            vx_ball = vx_ball
            y_rel = y_rel + self.tau * vy_ball
            vy_ball = vy_ball - self.g * self.tau
        else:
            raise Exception("Integrator not recognized.")

        self.state = (x_rel, vx_ball, y_rel, vy_ball)

        # Check if we have gone beyond time horizon
        if (self.t > (self.T-1)):
            self.steps_beyond_done = self.t - (self.T-1)
            done = True
        else:
            done = False

        if done: # done only if beyond time horizon
            reward = 0.0
        else:
            reward = min(x_rel/self.reward_thresh+1, -x_rel/self.reward_thresh+1)
            reward = max(reward, 0.0)


        obs_with_noise = self.add_noise(np.array(self.state))

        # Kalman filter
        if self.kalman_filter:
            self.filter.predict(u=np.array([[u],[self.g]]))
            self.filter.update(obs_with_noise)
            state_estimate = np.reshape(self.filter.x, (self.nx,))
            obs_with_noise = state_estimate

        # Update time
        self.t += 1

        return obs_with_noise, reward, done, {}

    def reset(self):
        self.t = 0

        # Gaussian distribution
        self.state = self.np_random.normal(self.x0_mu, self.x0_belief_std_dev)
        

        # Generate observation
        self.steps_beyond_done = None
        obs_w_noise = self.add_noise(np.array(self.state))

        # Reset filter
        if self.kalman_filter:
            self.filter.x = self.initial_state_estimate
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


    def integrate_linear_gaussian(self, c, d, mu, sigma, x):
        # integrate (c*x+d)*gaussian_pdf(mu,sigma) dx
        # gives indefinite integral, which is then evaluated at x
        # computed using Wolfram Alpha:
        # int (c*x+d)*PDF[NormalDistribution[mu,sigma],x] dx
        return 0.5*(c*mu+d)*math.erf((x-mu)/(np.sqrt(2)*sigma)) - c*(sigma**2)*stats.norm.pdf(x,loc=mu,scale=sigma)

    def compute_expected_reward(self, mu_x, cov_x):
        # Compute expected reward using Gaussian distribution of state

        # First, compute marginal distribution of x_rel (first component of state)
        mu = mu_x[0][0]
        sigma = np.sqrt(cov_x[0][0])

        reward_thresh = self.reward_thresh

        c1 = 1/reward_thresh; d1 = 1
        expected_reward1 = self.integrate_linear_gaussian(c1,d1,mu,sigma,0) - self.integrate_linear_gaussian(c1,d1,mu,sigma,-reward_thresh)

        c2 = -1/reward_thresh; d2 = 1
        expected_reward2 = self.integrate_linear_gaussian(c2,d2,mu,sigma,reward_thresh) - self.integrate_linear_gaussian(c2,d2,mu,sigma,0)

        expected_reward = expected_reward1 + expected_reward2

        return expected_reward

    def state_distribution(self, u_seq):
        # Propagate state distribution forwards using u_seq
        H = len(u_seq) # length of input sequence
        mu_state = self.x0_mu.T
        mu_state = mu_state.reshape((self.nx,1))
        cov_state = self.x0_cov 
        A = self.A_transition
        for k in range(H):
            action = u_seq[k]
            u = self.inputs[action]
            v = np.array([[-self.tau*u, 0, 0, -self.tau*self.g]]).T
            mu_state = A @ mu_state + v
            cov_state = A @ cov_state @ A.T

        return mu_state, cov_state

    def compute_MI(self, mu_x, cov_x):
        # Compute mutual information given state distribution

        # Conditional distribution of measurements: y|x ~ N(x, sigma_y_x^2)  
        mu_y = mu_x # mean of marginal
        sigma_y_x = self.noise_std_dev
        cov_y_x = np.diag(self.noise_std_dev**2)

        # Marginal distribution of y
        # See Eq. (2.105) here: https://www.seas.upenn.edu/~cis520/papers/Bishop_2.3.pdf
        cov_y = cov_y_x + cov_x

        # Covariance matrix of joint distribution over (x,y)
        cov_xy = np.block([
            [cov_x, cov_x],
            [cov_x, cov_y]
            ])

        # Exact MI; see https://www.math.nyu.edu/~kleeman/infolect7.pdf
        # Assumes mean of y is same as mean of x (which is true here)
        exact_MI = 0.5*np.log(np.linalg.det(cov_x)*np.linalg.det(cov_y)/np.linalg.det(cov_xy))

        return exact_MI

    def MPC(self, u_seqs):
        
        # Initialize best expected reward
        best_reward = -np.inf

        # Iterate over all input sequences
        for i in range(len(u_seqs)):
            u_seq = u_seqs[i]
            reward_u_seq = self.expected_rewards_u_seq(u_seq)
            if reward_u_seq > best_reward:
                best_reward = reward_u_seq
                best_u_seq = u_seq

        # Apply first action in best sequence
        action = best_u_seq[0]

        return action

    def expected_rewards_u_seq(self, u_seq):

        # Propagate state distribution forwards using u_seq and compute expected rewards
        H = len(u_seq) # length of input sequence
        mu_state = np.reshape(self.filter.x, (self.nx,1))
        cov_state = self.filter.P
        A = self.A_transition
        reward_u_seq = 0.0
        for k in range(H):
            action = u_seq[k]
            u = self.inputs[action]
            v = np.array([[-self.tau*u, 0, 0, -self.tau*self.g]]).T
            mu_state = A @ mu_state + v
            cov_state = A @ cov_state @ A.T
            reward = self.compute_expected_reward(mu_state, cov_state)
            reward_u_seq += reward

        return reward_u_seq