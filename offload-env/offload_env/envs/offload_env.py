from stochastic_timeseries_facenet import facenet_stochastic_video
import gym
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

# Includes the additional features:
# 	- 'past_local_input_x_diff' which is the difference 
# 	   between the current input x and the last input on 
# 	   which the local model was queried
# 	- 'past_cloud_input_x_diff' which is the difference 
#	   between the current input x and the last input on
#	   which the cloud model was queried.
STATE_SPACE_FEATURES_3 = ['curr_input_x_diff',
						'past_local_input_x_diff',
						'past_cloud_input_x_diff',
						'past_local_predict',
						'past_cloud_predict',
						'past_local_tdiff',
						'past_cloud_tdiff',
						'num_cloud_queries_left',
						'time_left']

## Utils Functions:
###################

def state_dict_to_state_vec(state_dict, state_feature_list):
	state_vec = []
	for key in state_feature_list:
		if key == 'curr_input_x_embeddings':
			# 'curr_input_x_embeddings' maps to a 
			# np.array of SVM embeddings.
			embeddings_list = list(state_dict[key])
			state_vec += embeddings_list
		else:
			# All other state space features are numeric values.
			state_vec.append(state_dict[key])

def get_initial_state(curr_input_x_features, past_local_predict, past_local_confidence, 
						past_cloud_predict, num_cloud_queries_left):
	state_dict = {}
	
	# np.array of SVM embeddings for input x.
	state_dict['curr_input_x_features'] = curr_input_x_features		
	# Difference between SVM embeddings for curr input x 
	# and input at previous timestep.
	state_dict['curr_input_x_diff'] = 0.0					# 

	# Local model state information.
	state_dict['past_local_input_x'] = curr_input_x_features
	state_dict['past_local_predict'] = past_local_predict
	state_dict['past_local_conf'] = past_local_confidence
	state_dict['past_local_input_x_diff'] = 0.0
	state_dict['past_local_tdiff'] = 0

	# Cloud model state information.
	state_dict['past_cloud_input_x'] = curr_input_x_features
	state_dict['past_cloud_predict'] = past_cloud_predict
	state_dict['past_cloud_tdiff'] = 0
	state_dict['past_cloud_input_x_diff'] = 0.0
	state_dict['num_cloud_queries_left'] = num_cloud_queries_left

	# Tuple contains the final prediction and corresponding confidence,
	# which is dependent on the action taken.
	state_dict['curr_chosen_prediction'] = (None, 0.0)

	# General sensory information.
	state_dict['time_left'] = self.T - 1

	return state_dict

def distance(curr_input_x_features, prev_input_x_features):
	diff = curr_input_x_features - prev_input_x_features
	return np.sqrt(np.sum(np.square(diff)))

class OffloadEnv(gym.Env):
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
		self.rewards_params_dict['weight_of_accuracy_cost'] = 10.0j

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

		self.CLOUD_CONF = 1.0 # Cloud confidence is assumed to be 100%.

		## State Space.
		######################
		self.n_s = len(state_space_features)

	"""
	 Implement dynamics: so given (s, a) return (s', reward, done, info).
	 Note that action will be numeric.
	"""
	def _step(self, action):
		nominal_action_name = self.numeric_to_action_dict[action]
		allowed_action_name = self.get_action_name(nominal_action_name)
		# Get numeric value of action we're allowed to take
		allowed_action_numeric = self.action_to_numeric_dict[allowed_action_name]

		# Update state dictionary info, that is action specifc, to the next state.
		self._propogateState(allowed_action_name)

		# Compute reward given the current state and action.
		true_output_y = self.timeseries_dict['true_value_vec'][self.t]
		reward = self._computeReward(allowed_action_numeric, true_output_y)

		# Update timestep, other sensory state info that is not action-specific.
		self.t += 1
		self.state_dict['time_left'] -= 1

		done_flag = False
		if self.t == self.T:
			done_flag = True

		curr_input_x_features = self.timeseries_dict['query_ts'][self.t]
		self.state_dict['curr_input_x_features'] = curr_input_x_features
		prev_input_x_features = self.timeseries_dict['query_ts'][self.t - 1]

		curr_input_x_diff = distance(curr_input_x_features, prev_input_x_features)
		self.state_dict['curr_input_x_diff'] = curr_input_x_diff

		past_local_input_x = self.state_dict['past_local_input_x']
		self.state_dict['past_local_input_x_diff'] = distance(curr_input_x_features, past_local_input_x)
		past_local_cloud_x = self.state_dict['past_cloud_input_x']
		self.state_dict['past_cloud_input_x_diff'] = distance(curr_input_x_features, past_local_cloud_x)

		self.state_dict['past_local_tdiff'] += 1
		self.state_dict['past_cloud_tdiff'] += 1

		# Return next state, reward, whether we are done, info dict.
		next_state_vec = state_dict_to_state_vec(self.state_dict, self.CURR_STATE_SPACE_FEATURES)
		return next_state_vec, reward, done_flag, {}

	def _propogateState(self, action_name):
		# Propogate to the next state given the action we're taking.
		if action_name == 'past_local':
			# Choose to use past local model prediction.
			previous_local_predict = self.state_dict['past_local_predict']
			previous_local_conf = self.state_dict['past_local_conf']

			self.state_dict['curr_chosen_prediction'] = (previous_local_predict, previous_local_conf)

		elif action_name == 'past_cloud':
			# Choose to use past cloud prediction.
			previous_cloud_predict = self.state_dict['past_cloud_predict']
			previous_cloud_conf = self.CLOUD_CONF

			self.state_dict['curr_chosen_prediction'] = (previous_cloud_predict, previous_cloud_conf)

		elif action_nam == 'query_local':
			# Choose to query the local model.
			local_prediction_vec = self.timeseries_dict['local_prediction_vec']
			local_confidence_vec = self.timeseries_dict['local_confidence_vec']

			curr_local_predict = local_prediction_vec[self.t]
			curr_local_conf = local_confidence_vec[self.t]

			self.state_dict['curr_chosen_prediction'] = (curr_local_predict, curr_local_conf)
			self.state_dict['past_local_predict'] = curr_local_predict
			self.state_dict['past_local_conf'] = curr_local_conf
			self.state_dict['past_local_input_x'] = self.state_dict['curr_input_x_features']
			self.state_dict['past_local_tdiff'] = 0

		elif action_name == 'query_cloud':
			# Choose to query the cloud model.
			cloud_prediction_vec = self.timeseries_dict['cloud_prediction_vec']

			curr_cloud_predict = cloud_prediction_vec[self.t]
			curr_cloud_conf = self.CLOUD_CONF

			self.state_dict['curr_chosen_prediction'] = (curr_cloud_predict, curr_cloud_conf)
			self.state_dict['past_cloud_predict'] = curr_cloud_predict
			self.state_dict['past_cloud_input_x'] = self.state_dict['curr_input_x_features']
			self.state_dict['past_cloud_tdiff'] = 0
			self.state_dict['num_cloud_queries_left'] -= 1

	def _computeReward(self, action_numeric, true_output_y):
		weight_of_query_cost = self.rewards_params_dict['weight_of_query_cost'] = 1.0
		weight_of_accuracy_cost = self.rewards_params_dict['weight_of_accuracy_cost'] = 10.0j

		query_cost = self.query_cost_dict[action_numeric]
		accuracy_cost = None

		prediction = self.state_dict['curr_chosen_prediction'][0]
		if prediction == true_output_y:
			accuracy_cost = 0.0
		else:
			accuracy_cost = 1.0

		reward = -1.0 * weight_of_accuracy_cost * accuracy_cost - weight_of_query_cost * query_cost 
		return reward
		
	def _render(self, mode='human', close=False):
        pass

	def _reset(self, coherence_time=8, P_SEEN=0.6, T=80, CURR_STATE_SPACE_FEATURES=STATE_SPACE_FEATURES_1, train_test = 'TRAIN'):
		# Reset timestep.
		self.t = 0
		self.SVM_results_df = pandas.read_csv('SVM_results.csv')

		# Select the state space definition to use.
		self.CURR_STATE_SPACE_FEATURES = CURR_STATE_SPACE_FEATURES

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
		self.local_confidence_vec, self.local_prediction_vec, self.cloud_prediction_vec, self.true_value_vec, self.edge_cloud_accuracy_gap_vec, self.query_ts, self.seen_vec, self.rolling_diff_vec, self.image_name_vec, self.train_test_membership_vec, self.embedding_norm_vec = facenet_stochastic_video(SVM_results_df=, T=T, coherence_time=coherence_time, P_SEEN=P_SEEN, train_test_membership=train_test)

		# Budget of queries for cloud model.
		self.num_cloud_queries_left = 10

		# Max Timestep in the sampled timeseries
		self.T = len(timeseries_dict['query_ts']) - 1

		# Initialize the initial state.
		# The current state values are maintained within a dictionary.
		self.state_dict = get_initial_state(curr_input_x_features=self.query_ts[self.t],
											past_local_predict=self.local_prediction_vec[self.t],
											past_local_confidence=self.local_confidence_vec[self.t],
											past_cloud_predict=self.cloud_prediction_vec[self.t],
											num_cloud_queries_left=self.num_cloud_queries_left)

		state = state_dict_to_state_vec(order_list=self.CURR_STATE_SPACE_FEATURES, state_dict=self.state_dict)

		return state






