#!/usr/bin/env python3

'''
Created on 7 Mar 2023
@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
import numpy as np

# This function calculates the mean squared error in order to form an accurate
# quantitative evaluation of mc policies performance
def calculate_mse_2d(predicted, actual):
    predicted = np.nan_to_num(predicted, nan=0.0)  # This replaces nan with 0 in predicted
    actual = np.nan_to_num(actual, nan=0.0)  # This replaces nan with 0 in actual
    mse = np.mean((predicted - actual) ** 2)
    return mse

if __name__ == '__main__':
    # This array holds a list of episodes to iterate over in order to 
    # experiment the effect they have
    episode_counts = [50, 100, 200]  
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)

    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    pe.evaluate()
    ground_truth_values = pe.value_function()._values  
    print("Ground Truth Values:", ground_truth_values)

    # For loop to iterate through each set of policy counts
    for episode_count in episode_counts:
        # On policy MC predictor
        mcpp = OnPolicyMCPredictor(env)
        mcpp.set_target_policy(pi)
        mcpp.set_experience_replay_buffer_size(64)
        mcpp.set_use_first_visit(True)

        # Off policy MC predictor
        mcop = OffPolicyMCPredictor(env)
        mcop.set_target_policy(pi)
        mcop.set_experience_replay_buffer_size(64)
        b = env.initial_policy()
        b.set_epsilon(0.2)
        mcop.set_behaviour_policy(b)
        mcop.set_use_first_visit(True)

        # Evaluate and update for each specified number of epsidoes
        for e in range(episode_count):
            mcpp.evaluate()
            mcop.evaluate()

        # Accessing the values directly from MC predictors to perform the MSE calculation
        mcpp_values = mcpp.value_function()._values
        mcop_values = mcop.value_function()._values

        # Calculate MSE
        mse_mcpp = calculate_mse_2d(mcpp_values, ground_truth_values)
        mse_mcop = calculate_mse_2d(mcop_values, ground_truth_values)

        print(f"MSE for MC-on with {episode_count} episodes: {mse_mcpp}")
        print(f"MSE for MC-off with {episode_count} episodes: {mse_mcop}")


    # v_pe.save_screenshot("q1_b_truth_pe.pdf")
    # v_mcop.save_screenshot("q1_b_mc-off_pe.pdf")
    # v_mcpp.save_screenshot("q1_b_mc-on_pe.pdf")
