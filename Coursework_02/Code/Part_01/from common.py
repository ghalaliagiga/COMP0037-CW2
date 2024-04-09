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
class TDPolicyPredictor(TDAlgorithmBase):

  class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment, alpha, gamma):
        super().__init__(environment)  # Assuming Python 3 syntax for simplicity
        
        self._alpha = alpha
        self._gamma = gamma
        self._minibatch_buffer = []
        # Initialize other necessary attributes here

                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            

    def _update_value_function_from_episode(self, episode):
    # Assume alpha and gamma are defined; if not, they need to be set as part of the class
     alpha = self._learning_rate
     gamma = self._discount_factor

    # Iterate through the episode, excluding the terminal state
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

        # Q1e:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        self._v.set_value(x_cell_coord, y_cell_coord, new_v)

        # Example to show how to extract coordinates; this does not do anything useful
        coords = episode.state(0).coords()
        new_v = 0
        self._v.set_value(coords[0], coords[1], new_v)
