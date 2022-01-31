from gym.envs.registration import register

register(
	id='BallCatching-v1',
	entry_point='gym_ball_catching.envs:BallCatchingEnv',
	)