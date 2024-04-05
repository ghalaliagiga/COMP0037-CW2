'''
Created on 7 Mar 2023

@author: steam
'''
import numpy as np
from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer


# This function calculates the mean squared error beteween two values and accounts
# for occurences of nan
def calculate_mse(predicted_values, true_values):
    predicted_values = np.nan_to_num(predicted_values, nan=0.0)
    true_values = np.nan_to_num(true_values, nan=0.0)
    mse = np.mean((predicted_values - true_values) ** 2)
    return mse

if __name__ == '__main__':
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)
    
    # Evaluate ground truth value function
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    pe.evaluate()
    ground_truth_values = pe.value_function()._values
    
    # Visualization for ground truth
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)
    v_pe.draw()
    v_pe.save_screenshot("ground_truth_value_function.pdf")

    epsilon_b_values = [0.1, 0.2, 0.5, 1.0]
    for epsilon_b in epsilon_b_values:
        b = env.initial_policy()
        b.set_epsilon(epsilon_b)
        
        mc_predictor = OffPolicyMCPredictor(env)
        mc_predictor.set_target_policy(pi)
        mc_predictor.set_behaviour_policy(b)
        mc_predictor.set_use_first_visit(True)
        mc_predictor.set_experience_replay_buffer_size(64)

        # Evaluate and update value function for 100 episodes
        for _ in range(100):
            mc_predictor.evaluate()

        # Calculate and print MSE
        predicted_values = mc_predictor.value_function()._values
        mse = calculate_mse(predicted_values, ground_truth_values)
        print(f"MSE for epsilon_b={epsilon_b:.2f}: {mse}")

        # Visualize and save predicted value function for current epsilon_b
        '''v_mc = ValueFunctionDrawer(mc_predictor.value_function(), drawer_height)
        v_mc.draw()
        v_mc.save_screenshot(f"mc_off_epsilon_b_{epsilon_b:.2f}.pdf")'''