# Create list representation of state space,
# features as defined in equation (3). Note
# that because our initial state is always the
# same, and the step function returns s'
# this doesn't need to be defined within the
# class below.
state_space_features = ['input_x_features', 
						'past_local_predict', 
						'past_cloud_predict', 
						'past_local_tdiff', 
						'past_cloud_tdiff', 
						'num_queries_left',  
						'time_left']

class OffloadEnv:
	"""
		Initialize the environment.
	"""

	def __init__(self):
		## Reward Parameters.
		######################
		# These are the alpha and beta parameters that are 
		## used in the calculation of the reward function defined as 
		## equation (4) in Sandeep's paper.
		self.rewards_params_dict = {}
		self.rewards_params_dict['weight_of_query_cost'] = 1.0
		self.rewards_params_dict['weight_of_accuracy_cost'] = 10.0

		# These are the costs of the four available queries
		# The first two will always have cost zero because they
		# re-use past query results, the next two are fixed 
		# values from the paper for now but can be parameterized later.
		self.query_cost_dict = {}
		self.query_cost_dict[0] = 0.0
		self.query_cost_dict[1] = 0.0
		self.query_cost_dict[2] = 1.0
		self.query_cost_dict[3] = 5.0

		# TODO: Check out if seed is necessary.

		## Action Space.
		######################
		# Create dictionaries for going from
		# numeric to word representation and vice versa
		self.action_to_numeric_dict = {}
		self.action_to_numeric_dict['past_local'] = 0
		self.action_to_numeric_dict['past_cloud'] = 1
		self.action_to_numeric_dict['query_local'] = 2
		self.action_to_numeric_dict['query_cloud'] = 3

		self.numeric_to_action_dict = ['past_local', 'past_cloud', 'query_local', 'query_cloud']
		self.n_a = len(numeric_to_action_dict)


		## State Space.
		######################
		self.n_s = len(state_space_features)

	"""
	 Implement dynamics: so given (s, a) return (s', reward, done).
	 Note that action will be numeric.
	"""
	def _step(self, action):
		nominal_action_name = self.numeric_to_action_dict[action]
		# TODO: implement _get_action_name
		allowed_action_name = self._get_action_name(nominal_action_name)

		# Get numeric value of action we're allowed to take
		allowed_action = self.action_to_numeric_dict[allowed_action_name]

		# Propogate to the next state given the action we're taking.

		# Compute reward given the current state and action.

		# Return next state, reward, whether we are done, info dict.


	def _reset(self):
		pass




