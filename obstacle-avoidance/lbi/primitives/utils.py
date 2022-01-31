import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
import IPython as ipy


def check_primitive_collision(env, obs_radius, primitive):
    '''
	env: (2,N) numpy array
	obs_radius: scalar (obstacle radius)
	primitive: (2,K) numpy array
	'''

    # Compute pairwise distances between points in env and primitive
    dists = scipy.spatial.distance.cdist(env.T, primitive.T)

    # If any distance is less than or equal to radius, we have collision
    collision = np.any(dists <= obs_radius)

    return collision


def get_best_primitive(env, prim_lib):
    '''
	Get primitive index with largest distance to obstacles.
	'''

    # Number of primitives in library
    num_primitives = np.shape(prim_lib)[1]

    # Largest distance
    largest_dist = -np.inf

    for i in range(num_primitives):

        primitive = prim_lib[:, i, :]

        # Compute minimum distance to obstacles
        dist = primitive_dist_to_obs(env, primitive)

        # Update best primitive
        if dist > largest_dist:
            largest_dist = dist
            best_primitive_ind = i

    return best_primitive_ind


def primitive_dist_to_obs(env, primitive):
    '''
	env: (2,N) numpy array
	primitive: (2,K) numpy array

	Output: minimum distance to centers of obstacles (ignores obstacle radii)
	'''

    # Compute pairwise distances between points in env and primitive
    dists = scipy.spatial.distance.cdist(env.T, primitive.T)

    # Compute minimum distance
    min_dist = np.min(dists)

    return min_dist


def get_primitive_dists(env, prim_lib):
    '''
	Get distance of each primitive to obstacles.
	'''

    # Number of primitives in library
    num_primitives = np.shape(prim_lib)[1]

    # Distances
    dists = np.zeros(num_primitives)

    for i in range(num_primitives):

        primitive = prim_lib[:, i, :]

        # Compute minimum distance to obstacles
        dists[i] = primitive_dist_to_obs(env, primitive)

    return dists


def get_free_primitives(env, obs_radius, prim_lib):
	'''
	Returns vector with hot-encoding of collision-free primitives.
	'''

	# Number of primitives in library
	num_primitives = np.shape(prim_lib)[1]

	# Free primitives (one means free; zero means collision)
	free_primitives = np.zeros(num_primitives)

	for i in range(num_primitives):

		primitive = prim_lib[:, i, :]

		# Check collision
		collision = check_primitive_collision(env, obs_radius, primitive)

		free_primitives[i] = (not collision)

	return free_primitives





def plot_primitive(primitive):
    '''
	Plot motion primitive
	primitive: (2,K) numpy array
	'''

    plt.scatter(primitive[0, :], primitive[1, :])

    return None
