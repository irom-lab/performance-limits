import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
from lbi.primitives.utils import *
import IPython as ipy


def generate_random_env(xlims, ylims, num_obs):
    '''
	Generate random environment with x-y locations of num_obs obstacles in xlims and ylims.
	'''

    # Generate x locations
    x_locations = np.random.rand(1, num_obs)
    x_locations = (xlims[1] - xlims[0]) * x_locations + xlims[0]

    # Generate y locations
    y_locations = np.random.rand(1, num_obs)
    y_locations = (ylims[1] - ylims[0]) * y_locations + ylims[0]

    # Put them together
    env = np.concatenate((x_locations, y_locations))

    return env

def random_traversable_env(xlims, ylims, num_obs, obs_radius, prim_lib):
    '''
    Generate random environment with x-y locations of num_obs obstacles in xlims and ylims 
    that is traversable using primitive library.
    '''

    traversable = False

    num_primitives = np.shape(prim_lib)[1]

    while not traversable:

        # Generate random env
        env = generate_random_env(xlims, ylims, num_obs)

        # Check if env is traversable
        for k in range(num_primitives):
            primitive = prim_lib[:,k,:]
            collision = check_primitive_collision(env, obs_radius, primitive)
            if not collision:
                traversable = True
                break

    return env

def random_traversable_env_pool(inputs):
    '''
    Parallelizable version.
    Generate random environment with x-y locations of num_obs obstacles in xlims and ylims 
    that is traversable using primitive library.
    '''

    # Unpack inputs
    xlims = inputs[0][0]
    ylims = inputs[1][0]
    num_obs = inputs[2][0]
    obs_radius = inputs[3][0]
    prim_lib = inputs[4][0]

    traversable = False

    num_primitives = np.shape(prim_lib)[1]

    while not traversable:

        # Generate random env
        env = generate_random_env(xlims, ylims, num_obs)

        # Check if env is traversable
        for k in range(num_primitives):
            primitive = prim_lib[:,k,:]
            collision = check_primitive_collision(env, obs_radius, primitive)
            if not collision:
                traversable = True
                break

    return env



def plot_env(env, obs_radius, x_lims, y_lims):
    '''
	Plot environment
	env: (2,N) numpy array
	obs_radius: scalar
	'''

    num_obs = env.shape[1]

    fig, ax = plt.subplots()

    for i in range(num_obs):
        circle = plt.Circle(env[:, i], obs_radius, color='r')
        ax.add_patch(circle)

    plt.xlim(x_lims)
    plt.ylim(y_lims)
    ax.axis('equal')
    # plt.show()
