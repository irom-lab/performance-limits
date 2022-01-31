import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import IPython as ipy
import json
import lbi.geom as geom
import time
from lbi.envs.envs import *


class Environment:

    def __init__(self, obs_radius, num_obs, obs_locations, parallel=False, gui=True, obs_alpha=1):
        self.parallel = parallel
        self.gui = gui

        self.height_obs = 3
        self.robot_height = 0.75
        self.shift_plot = [-5, 2.0]
        self.robot_radius = 0.3
        self.obs_radius = obs_radius
        self.num_obs = num_obs
        self.posObs = None
        self.radObs = None
        self.obs_locations = obs_locations
        self.obs_alpha = obs_alpha # Transparency of obstacles

        self.p = None
        self.husky = None
        self.sphere = None
        self.setup_pybullet()

    def setup_pybullet(self):

        if self.parallel:
            if self.gui:
                p = bc.BulletClient(connection_mode=pybullet.GUI)
            else:
                p = bc.BulletClient(connection_mode=pybullet.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            if self.gui:
                pybullet.connect(pybullet.GUI)
                p = pybullet
                # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
            else:
                pybullet.connect(pybullet.DIRECT)
                p = pybullet

        self.ground = p.loadURDF("visualization/URDFs/plane.urdf") # ,globalScaling=5.0)  # Ground plane
        p.changeVisualShape(self.ground, -1, rgbaColor=[0.9,0.9,0.9,1.0])
        self.init_position = [self.shift_plot[0], self.shift_plot[1], self.robot_height]
        self.init_orientation = p.getQuaternionFromEuler([0., 0., np.pi/4])
        quadrotor = p.loadURDF("visualization/URDFs/quadrotor.urdf", 
                               basePosition=self.init_position,
                               baseOrientation=self.init_orientation,
                               useFixedBase=1, 
                               globalScaling=0.5)  # Load robot from URDF
        p.changeVisualShape(quadrotor, -1, rgbaColor=[0.5,0.5,0.5,1])
        
        self.p = p
        self.quadrotor = quadrotor

    def set_gui(self, gui):
        self.p.disconnect()
        self.gui = gui
        self.setup_pybullet()

    def generate_obstacles(self):
        p = self.p
        numObs = self.num_obs

        linkMasses = [None] * (numObs)
        colIdxs = [None] * (numObs)
        visIdxs = [None] * (numObs)
        posObs = [None] * (numObs)
        radObs = [None] * (numObs)
        orientObs = [None] * (numObs)
        parentIdxs = [None] * (numObs)
        linkInertialFramePositions = [None] * (numObs)
        linkInertialFrameOrientations = [None] * (numObs)
        linkJointTypes = [None] * (numObs)
        linkJointAxis = [None] * (numObs)

        for obs in range(numObs):
            linkMasses[obs] = 0.0
            visIdxs[obs] = -1
            parentIdxs[obs] = 0
            linkInertialFramePositions[obs] = [0, 0, 0]
            linkInertialFrameOrientations[obs] = [0, 0, 0, 1]
            linkJointTypes[obs] = p.JOINT_FIXED
            linkJointAxis[obs] = np.array([0, 0, 1])
            orientObs[obs] = [0, 0, 0, 1]

        posObs, colIdxs, visIdxs, radObs = self.generate_obstacles_sub(p, posObs, colIdxs, visIdxs, radObs)
        
        obsUid = p.createMultiBody(baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[0, 0, 0],
                                   baseOrientation=[0, 0, 0, 1], baseInertialFramePosition=[0, 0, 0],
                                   baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
                                   linkCollisionShapeIndices=colIdxs, linkVisualShapeIndices=visIdxs,
                                   linkPositions=posObs, linkOrientations=orientObs, linkParentIndices=parentIdxs,
                                   linkInertialFramePositions=linkInertialFramePositions,
                                   linkInertialFrameOrientations=linkInertialFrameOrientations,
                                   linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis)
        self.posObs = posObs
        self.radObs = radObs

        return obsUid

    def generate_obstacles_sub(self, p, posObs, colIdxs, visIdxs, radObs):
        numObs = self.num_obs
        heightObs = self.height_obs
        
        color_lib = [[0.8,0.8,0,self.obs_alpha], [0,0.7,0,self.obs_alpha], [0,0.5,1,self.obs_alpha], [0.7,0,0,self.obs_alpha]]
        

        for obs in range(numObs): # Cylindrical obstacles
            posObs_obs = np.array([None] * 3)
            radiusObs = self.obs_radius
            
            posObs_obs[0] = self.obs_locations[0,obs] + self.shift_plot[0]
            posObs_obs[1] = self.obs_locations[1,obs] + self.shift_plot[1]
            posObs_obs[2] = 0.0  # set z at ground level
            posObs[obs] = posObs_obs
            radObs[obs] = radiusObs
            # radiusObs = rmin + (rmax - rmin) * np.random.random_sample(1)
            colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                  radius=radiusObs, 
                                                  height=heightObs)

            cylinder_color = color_lib[np.random.randint(low=0, high=4)]
            visIdxs[obs] = p.createVisualShape(pybullet.GEOM_CYLINDER, 
                                               radius=radiusObs, 
                                               length=heightObs,
                                               rgbaColor=cylinder_color)

        return posObs, colIdxs, visIdxs, radObs

    def plot_lidar(self, num_rays, lidar_angle_range, lidar_max_distance):

        p = self.p

        thetas = np.linspace(lidar_angle_range[0], lidar_angle_range[1], num_rays)
        thetas = thetas.reshape(num_rays,1)

        raysTo = np.concatenate((self.shift_plot[0]+lidar_max_distance*np.cos(thetas), self.shift_plot[1]+lidar_max_distance*np.sin(thetas), self.robot_height*np.ones((num_rays,1))), 1)


        for k in range(num_rays):
            p.addUserDebugLine(np.array([self.shift_plot[0],self.shift_plot[1],self.robot_height]), raysTo[k,:], lineColorRGB = [1,0,0], lineWidth=2, lifeTime=0)

    def plot_prim_lib(self, prim_lib):


        p = self.p
        num_primitives = np.shape(prim_lib)[1]

        for k in range(num_primitives):

            primitive = prim_lib[:,k,:]

            for t in range(primitive.shape[1]-1):
                p.addUserDebugLine(np.array([self.shift_plot[0]+primitive[0,t],self.shift_plot[1]+primitive[1,t],self.robot_height]), np.array([self.shift_plot[0]+primitive[0,t+1],self.shift_plot[1]+primitive[1,t+1],self.robot_height]), lineColorRGB = [0,0,1], lineWidth=2, lifeTime=0)



###################################################################
# Load params from json file
with open("params.json", "r") as read_file:
    params = json.load(read_file)


# Define workspace
workspace_x_lims = params["workspace_x_lims"]
workspace_y_lims = params["workspace_y_lims"]

# Robot state
robot_location = params["robot_location"]
robot_state = geom.Point(robot_location[0], robot_location[1])

# Limits for obstacles centers
obs_x_lims = params["obs_x_lims"]
obs_y_lims = params["obs_y_lims"]
num_obs = params["num_obs"]
obs_radius = params["obs_radius"]

# Define sensor
num_rays = params["num_rays"]
lidar_angle_range = params["lidar_angle_range"]
lidar_noise_std = params["lidar_noise_std"]
lidar_failure_rate = params["lidar_failure_rate"]
lidar_max_distance = params["lidar_max_distance"]

# Load motion primitives
prim_lib = np.load('lbi/primitives/primitive_library.npy')
num_primitives = np.shape(prim_lib)[1]
##################################################

##################################################
# Visualize environment
plot_lidar = True
plot_primitives = False
##################################################


# Obstacle locations
np.random.seed(8) # 5, 8, 12
obs_locations = random_traversable_env(obs_x_lims, obs_y_lims, num_obs, obs_radius, prim_lib)


if plot_lidar: 
    env = Environment(obs_radius, num_obs, obs_locations, obs_alpha=0.99, parallel=False, gui=True)
    env.generate_obstacles()
    env.plot_lidar(num_rays, lidar_angle_range, lidar_max_distance)
if plot_primitives:
    env = Environment(obs_radius, num_obs, obs_locations, obs_alpha=0.5, parallel=False, gui=True)
    env.generate_obstacles()
    env.plot_prim_lib(prim_lib)

# Set camera pose
env.p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=30.0, cameraPitch=-40.0, cameraTargetPosition=[-5,2,1.5])
time.sleep(120)
env.p.disconnect()