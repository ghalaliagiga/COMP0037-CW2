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

def _update_value_function_from_episode(self, episode):
    alpha = self._learning_rate
    gamma = self._discount_factor

    # Iterate through episodes, excluding optimal state
    for i in range(len(episode) - 1):
        current_state, action, reward, next_state = episode[i]
        
        # TD(0) Update for the value function
        # V(S) = V(S) + alpha * (reward + gamma * V(S') - V(S))
        current_value = self._v.get_value(current_state)
        next_value = self._v.get_value(next_state)
        td_target = reward + gamma * next_value
        td_error = td_target - current_value
        new_value = current_value + alpha * td_error
        
        # Update the value function for the current state
        self._v.set_value(current_state, new_value)

    # Example to show how to extract coordinates; this does not do anything useful
    # coords = episode.state(0).coords()
    # self._v.set_value(coords[0], coords[1], new_v) # Assuming new_v should be replaced or defined

