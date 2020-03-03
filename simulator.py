
# Note: we can experiment with different features 
# 		(that is, Phi(x)) of the problem sensory input x.
# 		so we create multiple state space feature lists.	

# This state space uses the SVM feature encoding from the database
# of the current input, as the sensory input features.
STATE_SPACE_FEATURES_1 = ['curr_input_x_embeddings', 
						'past_local_predict', 
						'past_cloud_predict', 
						'past_local_tdiff', 
						'past_cloud_tdiff', 
						'num_cloud_queries_left',  
						'time_left']

# This state space uses the vector difference between the 
# SVM feature embeddings of the current input x_t and the 
# SVM feature embeddings of the previous input x_t-1.
STATE_SPACE_FEATURES_2 = ['curr_input_x_diff', 
						'past_local_predict', 
						'past_cloud_predict', 
						'past_local_tdiff', 
						'past_cloud_tdiff', 
						'num_cloud_queries_left',  
						'time_left']

## Utils Functions:

def state_dict_to_state_vec(state_dict, order_list):
	state_vec = []
	for key in order_list:
		if key == 'curr_input_x_embeddings':
			# 'curr_input_x_embeddings' maps to a 
			# np.array of SVM embeddings.
			embeddings_list = list(state_dict[key])
			state_vec += embeddings_list
		else:
			# All other state space features are numeric values.
			state_vec.append(state_dict[key])

def get_initial_state(curr_input_x_features, past_local_predict, past_local_confidence
					  past_cloud_predict, num_cloud_queries_left):
	state_dict = {}
	
	# np.array of SVM embeddings for input x.
	state_dict['curr_input_x_features'] = curr_input_x_features		
	# Difference between SVM embeddings for curr input x 
	# and input at previous timestep.
	state_dict['curr_input_x_diff'] = 0.0					# 

	state_dict['past_local_predict'] = past_local_predict
	# state_dict['past_local_conf'] = past_local_confidence

	state_dict['past_cloud_predict'] = past_cloud_predict

	state_dict['past_local_tdiff'] = 0
	state_dict['past_cloud_tdiff'] = 0
	state_dict['num_cloud_queries_left'] = num_cloud_queries_left

	return state_dict

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
		self.query_cost_dict[0] = 0.0	# past local
		self.query_cost_dict[1] = 0.0	# past cloud
		self.query_cost_dict[2] = 1.0	# query local
		self.query_cost_dict[3] = 5.0 	# query cloud

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
			past_
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


	def _reset(self, coherence_time=8, P_SEEN=0.6, T=80, CURR_STATE_SPACE_FEATURES=STATE_SPACE_FEATURES_1):
		# Reset timestep.
		self.t = 0

		# Sample a new timeseries episode.
		# timeseries is a dict containing all the relevant info
		# about the time series:
		#
		# - local_prediction_vec: [prediction(x_1), ..., prediction(x_T)]
		#	 	List of class prediction numerics from the local model.
		# - local_confidence_vec: [conf(prediction(x_1)), ..., conf(prediction(x_T))]
		#		List of corresponding confidences to the local model predictions.
		# - cloud_prediction_vec: [prediction(x_1), ..., prediction(x_T)]
		#		List of class prediction numerics from the cloud model.
		#     	Note: the confidence of the cloud model is assumed to be 100%.
		# - query_ts: [phi(x_1), ..., phi(x_T)] 
		# 		List of input SVM feature embeddings for each input at 
		# 		each timestep, over timeseries.
		# - true_value_vec: 
		#		List of true class label numerics.
		# - edge_cloud_accuracy_gap_vec
		# - seen_vec
		# - rolling_diff_vec
		# - image_name_vec
		# - train_test_membership
		# - embeding_norm_vec
		timeseries_dict = facenet_stochastic_video(SVM_results_df=self.SVM_results_df, 
												   T=T, 
												   coherence_time=coherence_time, 
												   P_SEEN=P_SEEN, 
											  	   train_test_membership=train_test)

		# Budget of queries for cloud model.
		self.query_budget = 10
		self.num_cloud_queries_left = self.query_budget

		# Max Timestep in the sampled timeseries
		self.T = len(timeseries['query_ts'])

		# Initialize the initial state.
		# The current state values are maintained within a dictionary.
		self.state_dict = get_initial_state(curr_input_x_features=self.query_ts[self.t],
											past_local_predict=self.local_prediction_vec[self.t],
											past_local_confidence=self.local_confidence_vec[self.t],
											past_cloud_predict=self.cloud_prediction_vec[self.t],
											num_cloud_queries_left=self.num_cloud_queries_left)

		state = state_dict_to_state_vec(order_list=CURR_STATE_SPACE_FEATURES, state_dict=self.state_dict)

		return state






