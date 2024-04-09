#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

import time
import numpy as np
from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    
    # Select the controller
    policy_learner = QLearner(env)   
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)
    
    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
    mins = np.full(40, float('inf'))
    maxs = np.full(40, 0)
    means = np.full(40, float(0))
    iterations = 100
    for x in range(iterations):
        times = []
        for i in range(40):
            #print(i)
            start_time = time.time()
            policy_learner.find_policy()
            value_function_drawer.update()
            greedy_optimal_policy_drawer.update()
            pi.set_epsilon(1/math.sqrt(1+0.25*i))
            end_time = time.time()
            t = end_time - start_time
            times.append(t)
            means[i] += t
            if t > maxs[i]:
                maxs[i] = t
            if t < mins[i]:
                mins[i] = t
            #print("time = ", t)
            #print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")
        #print(times)
    
    means /= float(iterations)
    ranges = []
    for x in range(len(mins)):
        ranges.append(maxs[x] - mins[x])

    print("means: ", means)
    print("ranges: ", ranges)
        