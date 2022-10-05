"""
Lava Problem POMDP

#####################
#   #   # G #   # L #
#####################

"""

import pomdp_py
import random
import numpy as np
import sys
import IPython as ipy
import argparse

class LavaState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, LavaState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "LavaState(%s)" % self.name


class LavaAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, LavaAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "LavaAction(%s)" % self.name


class LavaObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, LavaObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "LavaObservation(%s)" % self.name


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, p_correct, num_states):
        self.p_correct = p_correct
        self.num_states = num_states # num_states # Number of actual states (ignoring dummy initial state)


    def probability(self, observation, next_state, action):

        if observation.name == "dummy": # we never observe dummy 
            return 0.0 
        elif next_state.name == "dummy": # we are in initial dummy state and the obs does not matter
            return 1/self.num_states
        else: # we are not in dummy state
            if next_state.name == observation.name: # observation = state (i.e., observation is correct)
                return self.p_correct
            else: # incorrect observation
                return (1-self.p_correct)/(self.num_states-1)

        return 0.0


    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [LavaObservation(s) for s in {"dummy", "0", "1", "2", "3", "4"}]


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, goal, lava, num_states):
        self.goal = goal
        self.lava = lava
        self.num_states= num_states

    def probability(self, next_state, state, action):
        """Lava and goal are absorbing states. Going left from left-most state returns the same state. 
        Otherwise, we move according to the action.
        There is also a dummy initial state. We start off in this state with probability one, and then
        transition to the real states at the next time step. This makes it so that we are consistent with
        the POMDP model we assume in the paper (where there is a state, then observation, then action); 
        this is different from the default in this pomdp_py package (where there is an action, followed by 
        a new state and observation, etc.). 
        """


        if next_state.name == "dummy": # we never transition to dummy
            return 0.0 

        if state.name == "dummy": # if we are in the initial dummy state
            if next_state.name is self.lava: # We don't start in lava
                return 0.0 
            else: # actual state is initialized uniformly in a non-lava state
                return 1/(self.num_states - 1) 

        else: # if we're not in the dummy initial state

            if (next_state.name==self.goal and state.name == self.goal): # the goal is absorbing
                return 1.0 
            if (next_state.name==self.lava and state.name == self.lava): # the lava is absorbing
                return 1.0 

            j = int(state.name)
            i = int(next_state.name)
            goal = int(self.goal)
            lava = int(self.lava)

            if (action.name.startswith("left") and max(j-1,0)==i and (j is not goal) and (j is not lava)): # i is left of j
                return 1.0 
            if (action.name.startswith("right") and min(j+1,self.num_states-1)==i and (j is not goal) and (j is not lava)): # i is right of j
                return 1.0 

        return 0.0

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [LavaState(s) for s in {"dummy", "0", "1", "2", "3", "4"}]


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, reward_x, goal, lava):
        self.reward_x = reward_x # reward for any non-goal/non-lava state
        self.goal = goal
        self.lava = lava

    def _reward_func(self, state, action):
        if state.name == "dummy":
            return 0.0 # no reward at dummy
        if state.name == self.lava:
            return 0.0 # no reward
        if state.name == self.goal:
            return 1.0 # reward

        return self.reward_x # reward for any other state

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely simple policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {LavaAction(s) for s in {"left", "right"}}

    def sample(self, state, **kwargs):
        return self.get_all_actions().random()


    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS



class LavaProblem(pomdp_py.POMDP):
    """
    Lava problem class.
    """

    def __init__(self, p_correct, reward_x, goal, lava, num_states, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(goal, lava, num_states),
                               ObservationModel(p_correct, num_states),
                               RewardModel(reward_x, goal, lava))
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(goal, lava, num_states),
                                   RewardModel(reward_x, goal, lava))
        super().__init__(agent, env, name="LavaProblem")


def main(raw_args=None):

    ##### Parse arguments #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_correct", type=float, default=0.2, help="probability of correct measurement (default: 0.2)")
    parser.add_argument("--reward_x", type=float, default=0.9, help="reward of being in non-goal/lava state (default: 0.9)")

    args = parser.parse_args(raw_args)
    p_correct = args.p_correct
    reward_x = args.reward_x

    ##### Other params #####
    horizon = 1 # Time horizon (without dummy state)
    goal = "2" # goal state 
    lava = "4" # lava state
    num_states = 5 # Number of actual states (ignoring dummy initial state)
    ########################


    # Initialize true state, belief, and lava problem instance
    init_true_state = LavaState("dummy") # We always start in dummy initial state
    init_belief = pomdp_py.Histogram({LavaState("dummy"): 1.0, LavaState("0"): 0.0, LavaState("1"): 0.0, LavaState("2"): 0.0, LavaState("3"): 0.0, LavaState("4"): 0.0})
    lava_problem = LavaProblem(p_correct, reward_x, goal, lava, num_states, init_true_state, init_belief)


    # Compute value associated with initial belief
    print("***Computing value via POMDP solution***")
    b = init_belief
    S = list(lava_problem.agent.all_states)
    A = list(lava_problem.agent.all_actions)
    Z = list(lava_problem.agent.all_observations)
    T = lava_problem.agent.transition_model  
    O = lava_problem.agent.observation_model  
    R = lava_problem.agent.reward_model  
    gamma = 1.0
    horizon_w_dummy= horizon + 1 # (extra time-step is due to initialization at dummy state)
    opt_value = pomdp_py.value(b, S, A, Z, T, O, R, gamma, horizon=horizon_w_dummy); 
    print("Optimal value: ", opt_value)
    opt_cost = horizon-opt_value
    print("Optimal cost: ", opt_cost)

    return opt_value

if __name__ == '__main__':
    main()

