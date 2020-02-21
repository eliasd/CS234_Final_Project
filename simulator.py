# Create list representation of state space,
# features as defined in equation (3). Note
# that because our initial state is always the
# same, and the step function returns s'
# this doesn't need to be defined within the
# class below.
STATE_SPACE_FEATURES = ['input_x_features', 
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

	def __init__(self, facenet_data_csv=FACENET_DATA_CSV):
		# Environment Data
		#####################
		self.SVM_results_df = pandas.read_csv(facenet_data_csv)

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
		if allowed_action_name == 'past_local':
			# TODO update prediction, update features (whatever we decide they should be)
		elif allowed_action_name == 'past_cloud':
			# TODO update prediction, update features, decrement queries left !
		elif allowed_action_name == 'query_local':
			# TODO update prediction, update features,
			# we want a vector with the local prediction, and the confidence of our prediction.
			# will look like this: curr_edge_prediction_vec = [self.edge_prediction_vec[self.t], self.edge_confidence_vec[self.t]]

			# update st
		elif allowed_action_name == 'query_cloud':


		# Compute reward given the current state and action.

		# Return next state, reward, whether we are done, info dict.


	def _reset(self):
		# Reset timestep.
		self.t = 0

		# Sample a new timeseries episode.
		# 
		timeseries = facenet_stochastic_video(SVM_results_df=self.SVM_results_df, 
											  T=T, 
											  coherence_time=coherence_time, 
											  P_SEEN=0.6, 
											  train_test_membership=train_test)

		# Budget of queries for cloud and edge model.
		self.query_budget = 10
		self.num_queries_remain = self.query_budget

		# Max Timestep in the sampled timeseries
		self.T = len(timeseries[query_ts])

		# Initialize the initial state.
		# The current state values are maintained within a dictionary.
		self.state_dict = get_initial_state(...)

		state = state_dict_to_state_vec(order_list=STATE_SPACE_FEATURES, state_dict=self.state_dict)
		return state






