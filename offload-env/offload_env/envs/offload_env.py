import gym
from gym.utils import seeding
from gym import spaces
import pandas as pd
import random
import numpy as np

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
						'time_left',
						'deadline']

# This state space uses the vector difference between the 
# SVM feature embeddings of the current input x_t and the 
# SVM feature embeddings of the previous input x_t-1.
STATE_SPACE_FEATURES_2 = ['curr_input_x_diff', 
						'past_local_predict', 
						'past_cloud_predict', 
						'past_local_tdiff', 
						'past_cloud_tdiff', 
						'num_cloud_queries_left',  
						'time_left',
						'deadline']

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
						'time_left',
						'deadline']

# TODO: implement dynamics in step and reset for the other
#		two state space types.
CURR_STATE_SPACE_FEATURES = STATE_SPACE_FEATURES_1
CURR_EPSIODE_LENGTH = 80
CURR_NUM_CLOUD_QUERIES_LEFT = 20
CURR_NUM_PREDICTION_LABELS = 10
EMBEDDING_DIM = 128


class OffloadEnv(gym.Env):
	"""
		Initialize the environment.
	"""

	def __init__(self):
		# Environment Data
		#####################
		self.SVM_results_df = pd.read_csv('SVM_results.csv')

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
		self.n_a = len(self.numeric_to_action_dict)

		self.CLOUD_CONF = 1.0 # Cloud confidence is assumed to be 100%.

		self.action_space = spaces.Discrete(self.n_a)

		## State Space.
		######################

		# For the deadline distribution
		# These values are equivalent to a distribution
		# with a mean of 0.5 and standard deviation of 0.2236
		self.d_alpha = 2
		self.d_beta = 2

		if CURR_STATE_SPACE_FEATURES == STATE_SPACE_FEATURES_1:
			'''
			self.observation_space = spaces.Dict({
				'PHI(input_x): Embeddings': spaces.Box(low=-np.inf, high=np.inf, shape = (128,), dtype= np.float64),
				'Past Local Prediction': spaces.Discrete(CURR_NUM_PREDICTION_LABELS),	# Range of the prediction space for dataset.
				'Past Cloud Prediction': spaces.Discrete(CURR_NUM_PREDICTION_LABELS),
				'Timesteps Since Last Local Query': spaces.Discrete(CURR_EPSIODE_LENGTH),
				'Timesteps Since Last Cloud Query': spaces.Discrete(CURR_EPSIODE_LENGTH),
				'Number of Cloud Model Queries Left': spaces.Discrete(CURR_NUM_CLOUD_QUERIES_LEFT),
				'Timesteps Left': spaces.Discrete(CURR_EPSIODE_LENGTH)
			})
			'''
			self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(128 + 1 + 1 + 1 + 1 + 1 + 1 + 1, ))
		elif CURR_STATE_SPACE_FEATURES == STATE_SPACE_FEATURES_2:
			'''
			self.observation_space = spaces.Dict({
				'PHI(input_x): Current Input Difference': spaces.Box(low=-np.inf, high=np.inf, shape = (1,), dtype= np.float64),
				'Past Local Prediction': spaces.Discrete(CURR_NUM_PREDICTION_LABELS),	# Range of the prediction space for dataset.
				'Past Cloud Prediction': spaces.Discrete(CURR_NUM_PREDICTION_LABELS),
				'Timesteps Since Last Local Query': spaces.Discrete(CURR_EPSIODE_LENGTH),
				'Timesteps Since Last Cloud Query': spaces.Discrete(CURR_EPSIODE_LENGTH),
				'Number of Cloud Model Queries Left': spaces.Discrete(CURR_NUM_CLOUD_QUERIES_LEFT),
				'Timesteps Left': spaces.Discrete(CURR_EPSIODE_LENGTH)
			})
			'''
		elif CURR_STATE_SPACE_FEATURES == STATE_SPACE_FEATURES_3:
			'''
			self.observation_space = spaces.Dict({
				'PHI(input_x)': spaces.Dict({
					'Current Input Difference': spaces.Box(low=-np.inf, high=np.inf, shape = (1,), dtype= np.float64),
					'Past Local Model Input Difference': spaces.Box(low=-np.inf, high=np.inf, shape = (1,), dtype= np.float64),
					'Past Cloud Model Input Difference': spaces.Box(low=-np.inf, high=np.inf, shape = (1,), dtype= np.float64)
				}),
				'Past Local Prediction': spaces.Discrete(CURR_NUM_PREDICTION_LABELS),	# Range of the prediction space for dataset.
				'Past Cloud Prediction': spaces.Discrete(CURR_NUM_PREDICTION_LABELS),
				'Timesteps Since Last Local Query': spaces.Discrete(CURR_EPSIODE_LENGTH),
				'Timesteps Since Last Cloud Query': spaces.Discrete(CURR_EPSIODE_LENGTH),
				'Number of Cloud Model Queries Left': spaces.Discrete(CURR_NUM_CLOUD_QUERIES_LEFT),
				'Timesteps Left': spaces.Discrete(CURR_EPSIODE_LENGTH)
			})
			'''
		else:
			# Should never get here. 
			print("--- CURR_STATE_SPACE_FEATURES is ill defined ---")

		## Seed.
		#####################
		# Default value
		self._seed = 22

		## Loss Logging.
		#################
		self.classification_error_avg = 0.0
		self.query_cost_avg = 0.0

	"""
	 Implement dynamics: so given (s, a) return (s', reward, done, info).
	 Note that action will be numeric.
	"""
	def step(self, action):
		nominal_action_name = self.numeric_to_action_dict[action]
		allowed_action_name = self.get_action_name(nominal_action_name)
		# Get numeric value of action we're allowed to take
		allowed_action_numeric = self.action_to_numeric_dict[allowed_action_name]

		# Update state dictionary info, that is action specifc, to the next state.
		self.propogateState(allowed_action_name)

		# Compute reward given the current state and action.
		true_output_y = self.timeseries_dict['true_value_vec'][self.t]
		reward = self.computeReward(allowed_action_numeric, true_output_y)

		# Update timestep to the next timestep, other sensory state info that is not action-specific.
		self.t += 1
		self.state_dict['time_left'] -= 1

		done_flag = False
		if self.t == self.T:
			#print("---- AVERAGE CLASSIFICATION ERROR: {}".format(self.classification_error_avg))
			#print("---- AVERAGE QUERY COST: {}".format(self.query_cost_avg))
			done_flag = True

		curr_input_x_embeddings = self.timeseries_dict['query_ts'][self.t]
		self.state_dict['curr_input_x_embeddings'] = curr_input_x_embeddings
		prev_input_x_features = self.timeseries_dict['query_ts'][self.t - 1]

		curr_input_x_diff = distance(curr_input_x_embeddings, prev_input_x_features)
		self.state_dict['curr_input_x_diff'] = curr_input_x_diff

		past_local_input_x = self.state_dict['past_local_input_x']
		self.state_dict['past_local_input_x_diff'] = distance(curr_input_x_embeddings, past_local_input_x)
		past_local_cloud_x = self.state_dict['past_cloud_input_x']
		self.state_dict['past_cloud_input_x_diff'] = distance(curr_input_x_embeddings, past_local_cloud_x)

		self.state_dict['past_local_tdiff'] += 1
		self.state_dict['past_cloud_tdiff'] += 1

		self.state_dict['deadline'] = np.random.beta(self.d_alpha, self.d_beta)

		# Return next state, reward, whether we are done, info dict.
		next_state_vec = state_dict_to_state_vec(self.state_dict, self.CURR_STATE_SPACE_FEATURES)
		return next_state_vec, reward, done_flag, {}

	def get_action_name(self, nominal_action_name):
		action_name = nominal_action_name

		if self.num_cloud_queries_left <= 0:
			if action_name == 'query_cloud':
				action_name = 'query_local'

		return action_name

	def propogateState(self, action_name):
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

		elif action_name == 'query_local':
			# Choose to query the local model.
			local_prediction_vec = self.timeseries_dict['local_prediction_vec']
			local_confidence_vec = self.timeseries_dict['local_confidence_vec']

			curr_local_predict = local_prediction_vec[self.t]
			curr_local_conf = local_confidence_vec[self.t]

			self.state_dict['curr_chosen_prediction'] = (curr_local_predict, curr_local_conf)
			self.state_dict['past_local_predict'] = curr_local_predict
			self.state_dict['past_local_conf'] = curr_local_conf
			self.state_dict['past_local_input_x'] = self.state_dict['curr_input_x_embeddings']
			self.state_dict['past_local_tdiff'] = 0

		elif action_name == 'query_cloud':
			# Choose to query the cloud model.
			cloud_prediction_vec = self.timeseries_dict['cloud_prediction_vec']

			curr_cloud_predict = cloud_prediction_vec[self.t]
			curr_cloud_conf = self.CLOUD_CONF

			self.state_dict['curr_chosen_prediction'] = (curr_cloud_predict, curr_cloud_conf)
			self.state_dict['past_cloud_predict'] = curr_cloud_predict
			self.state_dict['past_cloud_input_x'] = self.state_dict['curr_input_x_embeddings']
			self.state_dict['past_cloud_tdiff'] = 0
			self.state_dict['num_cloud_queries_left'] -= 1
			self.num_cloud_queries_left -= 1

	def computeReward(self, action_numeric, true_output_y):
		weight_of_query_cost = self.rewards_params_dict['weight_of_query_cost'] = 1.0
		weight_of_accuracy_cost = self.rewards_params_dict['weight_of_accuracy_cost'] = 10.0

		query_cost = self.query_cost_dict[action_numeric]
		accuracy_cost = None

		prediction = self.state_dict['curr_chosen_prediction'][0]
		if prediction == true_output_y:
			accuracy_cost = 0.0
		else:
			accuracy_cost = 1.0

		self.classification_error_avg += accuracy_cost * (1.0 / self.T)
		self.query_cost_avg += query_cost * (1.0 / self.T)

		reward = -1.0 * weight_of_accuracy_cost * accuracy_cost - weight_of_query_cost * query_cost - self.state_dict['deadline'] * query_cost
		return reward

	def reset(self, coherence_time=8, P_SEEN=0.6, train_test = 'TRAIN'):
		# Reset timestep.
		self.t = 0

		# Reset error logs.
		self.classification_error_avg = 0.0
		self.query_cost_avg = 0.0

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
		self.timeseries_dict = facenet_stochastic_video(SVM_results_df=self.SVM_results_df, T=CURR_EPSIODE_LENGTH, coherence_time=coherence_time, P_SEEN=P_SEEN, train_test_membership=train_test)

		# Budget of queries for cloud model.
		self.query_budget_frac = np.random.choice([0.10, 0.20, 0.50, 0.70, 1.0])
		self.num_cloud_queries_left = int(len(self.timeseries_dict['query_ts']) * self.query_budget_frac)

		# Max Timestep in the sampled timeseries
		self.T = len(self.timeseries_dict['query_ts']) - 1

		# Initialize the initial state.
		# The current state values are maintained within a dictionary.
		self.state_dict = get_initial_state(curr_input_x_embeddings=self.timeseries_dict['query_ts'][self.t],
											past_local_predict=self.timeseries_dict['local_prediction_vec'][self.t],
											past_local_confidence=self.timeseries_dict['local_confidence_vec'][self.t],
											past_cloud_predict=self.timeseries_dict['cloud_prediction_vec'][self.t],
											num_cloud_queries_left=self.num_cloud_queries_left,
											final_time_step=self.T,
											alpha=self.d_alpha,
											beta=self.d_beta)

		state = state_dict_to_state_vec(state_feature_list=self.CURR_STATE_SPACE_FEATURES, state_dict=self.state_dict)

		return state

	def render(self):
		pass


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

    return np.array(state_vec)

def get_initial_state(curr_input_x_embeddings, past_local_predict, past_local_confidence, 
                        past_cloud_predict, num_cloud_queries_left, final_time_step, alpha, beta):
    state_dict = {}
    
    # np.array of SVM embeddings for input x.
    state_dict['curr_input_x_embeddings'] = curr_input_x_embeddings
    # Difference between SVM embeddings for curr input x 
    # and input at previous timestep.
    state_dict['curr_input_x_diff'] = 0.0                   # 

    # Local model state information.
    state_dict['past_local_input_x'] = curr_input_x_embeddings
    state_dict['past_local_predict'] = past_local_predict
    state_dict['past_local_conf'] = past_local_confidence
    state_dict['past_local_input_x_diff'] = 0.0
    state_dict['past_local_tdiff'] = 0

    # Cloud model state information.
    state_dict['past_cloud_input_x'] = curr_input_x_embeddings
    state_dict['past_cloud_predict'] = past_cloud_predict
    state_dict['past_cloud_tdiff'] = 0
    state_dict['past_cloud_input_x_diff'] = 0.0
    state_dict['num_cloud_queries_left'] = num_cloud_queries_left

    # Tuple contains the final prediction and corresponding confidence,
    # which is dependent on the action taken.
    state_dict['curr_chosen_prediction'] = (None, 0.0)

    # General sensory information.
    state_dict['time_left'] = final_time_step - 1

    # Deadline (% of time passed towards deadline)
    state_dict['deadline'] = np.random.beta(alpha, beta)

    return state_dict

def distance(curr_input_x_embeddings, prev_input_x_features):
    diff = curr_input_x_embeddings - prev_input_x_features
    return np.sqrt(np.sum(np.square(diff)))

def get_random_uniform(p = 0.5):                                             
    random.seed()
    sample = np.random.uniform()                                             
    
    if sample <= p:                                                          
        return True                                                          
    else:
        return False    



def sample_specific_face(train_test_df = None, seen_boolean = None, sampled_face_label = None, seed = None):
   
    all_potential_faces_df = train_test_df[train_test_df.true_label_name == sampled_face_label]

    #np.random.seed(seed)

    num_rows = all_potential_faces_df.shape[0]

    random_row_idx = np.random.choice(range(num_rows))

    random_row_df = all_potential_faces_df.iloc[random_row_idx]

    image_name = random_row_df['image_id']

    embedding_vector = np.array([float(x) for x in random_row_df['embedding_vector'].split('_')])

    edge_cloud_accuracy_gap = 1.0 - random_row_df['model_correct']

    edge_prediction = random_row_df['SVM_prediction_numeric']
    
    edge_prediction_name = random_row_df['SVM_prediction']

    edge_confidence = random_row_df['SVM_confidence']

    cloud_prediction = random_row_df['true_label_numeric']

    cloud_prediction_name = random_row_df['true_label_name']

    train_test_membership = random_row_df['train_test_membership']

    return image_name, embedding_vector, edge_cloud_accuracy_gap, edge_prediction, edge_confidence, cloud_prediction, train_test_membership, edge_prediction_name, cloud_prediction_name

"""
    stochastic ts based on facenet
"""

    # how to create a stochastic timeseries
    # choose a coherence time
    # choose from EITHER train or test based on flag
    # based on P_SEEN, choose from either SEEN or UNKNOWN
    # for each coherence time, choose a random label from SEEN, UNKNOWN, labels
    # choose random images for that label for the coherence time
    # populate edge prediction, edge confidence, cloud and the gap timeseries for all of them
    # repeat until done
    
    # plot, SEEN, UNSEEN, confidence etc timeseries as before
    # run the all-edge, all-cloud, and rest of benchmarks for the AQE simulator
    # see how it does on facenet

def facenet_stochastic_video(SVM_results_df = None, T = 200, coherence_time = 10, P_SEEN = 0.7, print_mode = False, seed = None, train_test_membership = 'TRAIN', EMBEDDING_DIM = 128, mini_coherence_time = 3, POISSON_MODE = True):

    edge_confidence_vec = []
    edge_prediction_vec = []
    cloud_prediction_vec = []
    true_value_vec = []
    edge_cloud_accuracy_gap_vec = []
    input_vec = []
    seen_vec = []
    rolling_diff_vec = []
    image_name_vec = []
    train_test_membership_vec = []
    embedding_norm_vec = []

    np.random.seed(seed)

    train_test_df = SVM_results_df[SVM_results_df['train_test_membership'] == train_test_membership]
    
    if print_mode:
        print('SVM: ', SVM_results_df.shape)
        print('train_test: ', train_test_df.shape)

    # all faces seen in this df
    seen_face_names = list(set(train_test_df[train_test_df['seen_unseen'] == 'SEEN']['true_label_name']))
    unseen_face_names = list(set(train_test_df[train_test_df['seen_unseen'] == 'UNSEEN']['true_label_name']))

    if print_mode:
        print('seen_face_names: ', seen_face_names)
        print('unseen_face_names: ', unseen_face_names)

    zero_embedding_vector = np.zeros(EMBEDDING_DIM)
    past_embedding_vector = zero_embedding_vector

    if POISSON_MODE:
        empirical_coherence_time = np.random.poisson(coherence_time-1) + 1
    else:
        empirical_coherence_time = coherence_time

    for t in range(T):
        # generate properties for the distro
        if t % empirical_coherence_time == 0:
           
            seen = get_random_uniform(p = P_SEEN)
            
            # depending on seen or not, get the face from the appropriate bin
            if seen:
                # a sample face
                sampled_face_label = np.random.choice(seen_face_names)
            else:
                # a sample face
                sampled_face_label = np.random.choice(unseen_face_names)

            if print_mode:
                print('t: ', t, 'sampled_face: ', sampled_face_label, 'seen: ', seen)

            if POISSON_MODE:
                empirical_coherence_time = np.random.poisson(coherence_time-1) + 1
            else:
                empirical_coherence_time = coherence_time
            
            #print('empirical_coherence_time', empirical_coherence_time)

        # for this face, see the feasible images, embeddings, and confidences we can choose from
        
        if t % mini_coherence_time == 0:
            image_name, embedding_vector, edge_cloud_accuracy_gap, edge_prediction, edge_confidence, cloud_prediction, train_test_membership, edge_prediction_name, cloud_prediction_name = sample_specific_face(train_test_df = train_test_df, seen_boolean = seen, sampled_face_label = sampled_face_label, seed = seed)

        if print_mode:
            print(' ')
            print('seen: ', seen)
            print('sampled_label: ', sampled_face_label)
            print('edge_cloud_accuracy_gap: ', edge_cloud_accuracy_gap)
            print(' ')

        edge_confidence_vec.append(edge_confidence)
        input_vec.append(embedding_vector)
        edge_cloud_accuracy_gap_vec.append(edge_cloud_accuracy_gap)
        seen_vec.append(seen)
        train_test_membership_vec.append(train_test_membership)

        edge_prediction_vec.append(edge_prediction)
        cloud_prediction_vec.append(cloud_prediction)

        # ground-truth is cloud!!
        true_value_vec.append(cloud_prediction)

        rolling_diff = distance(embedding_vector, past_embedding_vector)
        #embedding_L2_norm = distance(embedding_vector, zero_embedding_vector) 
        embedding_L2_norm = np.mean([np.abs(x) for x in embedding_vector])

        # changed embedding vector
        past_embedding_vector = embedding_vector

        # new colums
        rolling_diff_vec.append(rolling_diff)
        image_name_vec.append(image_name)
        embedding_norm_vec.append(embedding_L2_norm)

    timeseries_dict = {}
    timeseries_dict['local_prediction_vec'] = edge_prediction_vec
    timeseries_dict['local_confidence_vec'] = edge_confidence_vec
    timeseries_dict['cloud_prediction_vec'] = cloud_prediction_vec
    timeseries_dict['query_ts'] = input_vec
    timeseries_dict['true_value_vec'] = true_value_vec
    timeseries_dict['edge_cloud_accuracy_gap_vec'] = edge_cloud_accuracy_gap_vec
    timeseries_dict['seen_vec'] = seen_vec
    timeseries_dict['rolling_diff_vec'] = rolling_diff_vec
    timeseries_dict['image_name_vec'] = image_name_vec
    timeseries_dict['train_test_membership'] = train_test_membership_vec
    timeseries_dict['embedding_norm_vec'] = embedding_norm_vec

    return timeseries_dict



