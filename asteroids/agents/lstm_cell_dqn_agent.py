import random
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, LeakyReLU, Dense, Dropout, LSTMCell
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import leaky_relu
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
from collections import deque

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from .reporting_callback import ReportingCallback
import time

NAME = 'asteroids-pilot-DuelingDQN-LSTMCells-{}'.format(int(time.time()))
EPOCHS_PER_TRAIN_STEP = 2
BATCH_SIZE = 256
TIMESPAN_LENGTH = 25

class SummaryNames:
    LEARNING_RATE = 'learning_rate'
    AVERAGE_SCORE = 'average_score'
    CHOICE_CONFIDENCE = 'choice_confidence'

class DQNAgent:
    def __init__(self, state_size, action_size, model=None, custom_summaries={}):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99    # discount rate
        self.learning_rate = 0.001
        self.action_probability_sharpening = 1
        self.action_probability_sharpening_increase = 0.02
        self.action_probability_sharpening_max = 2
        self.epoch_counter = 0
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()
        
        log_dir = 'logs/{}'.format(NAME)
        self.tensorboard = TensorBoard(log_dir=log_dir)
        self.checkpointer = ModelCheckpoint(filepath='models/{}.model'.format(NAME), verbose=1, save_best_only=True)

        self._this_game_reward = 0
        self._last_games_rewards = deque(maxlen=10)

        self._custom_summaries = custom_summaries
        custom_summary_names = self._custom_summaries.keys()
        self._reporting_callback = ReportingCallback(summary_names=[SummaryNames.LEARNING_RATE, SummaryNames.AVERAGE_SCORE, SummaryNames.CHOICE_CONFIDENCE] + list(custom_summary_names), tensorboard = self.tensorboard)

        keras_backend.set_learning_phase(0)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        observation_input = Input(shape=(self.state_size,))

        UNITS = 32
        cell_0_units = UNITS
        cell_1_units = UNITS
        cell_2_units = UNITS

        cell_0_state_in = [Input(shape=(cell_0_units,)) for _ in range(2)]
        cell_1_state_in = [Input(shape=(cell_1_units,)) for _ in range(2)]
        cell_2_state_in = [Input(shape=(cell_2_units,)) for _ in range(2)]

        output, (cell_0_output, cell_0_cell_state) = LSTMCell(cell_0_units, input_shape=(self.state_size,))(inputs=observation_input, states=cell_0_state_in)
        output = Dropout(rate=0.3)(output)
        output, (cell_1_output, cell_1_cell_state) = LSTMCell(cell_1_units, input_shape=(self.state_size,))(inputs=observation_input, states=cell_1_state_in)
        output = Dropout(rate=0.3)(output)
        #output, cell_2_state = LSTMCell(cell_2_units, input_shape=(self.state_size,))([observation_input, cell_2_state_in])
        #output = Dropout(rate=0.3)(output)

        value_estimator = Dense(1)(output)
        value_estimator = LeakyReLU()(value_estimator)

        advantage_estimator = Dense(self.action_size)(output)
        advantage_estimator = LeakyReLU()(advantage_estimator)

        quality_estimator = Lambda(lambda tensors: tensors[0] + tensors[1] - keras_backend.mean(tensors[1], axis=1, keepdims=True), output_shape=(self.action_size,))([value_estimator, advantage_estimator])
        
        model = Model(inputs=[observation_input] + cell_0_state_in + cell_1_state_in + cell_2_state_in, outputs=[quality_estimator, cell_0_output, cell_0_cell_state, cell_1_output, cell_1_cell_state])

        model.compile(loss=mean_squared_error,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    @staticmethod
    def sharpen(mantissa, exponents):
        """ like softmax, but with a custom mantissa """
        mantissa_raised_to_exponents = np.power(mantissa * np.ones_like(exponents), exponents - np.max(exponents))
        return mantissa_raised_to_exponents / mantissa_raised_to_exponents.sum()
        
    def act(self, state, user_data):
        stub_cell_input = [np.zeros([1,32])]*2
        batch_act_values = self.model.predict([state] + stub_cell_input + stub_cell_input + stub_cell_input)
        act_values = batch_act_values[0][0]

        assert(not np.any(np.isnan(act_values)))

        act_values_probabilities = DQNAgent.sharpen(self.action_probability_sharpening, act_values)
        
        action_selector_value = random.random()
        for action_index, action_probability in enumerate(act_values_probabilities):
            action_selector_value = action_selector_value - action_probability
            if action_selector_value <= 0:
                break

        assert(action_selector_value <= 0)
            
        return action_index, act_values[action_index], np.max(act_values), user_data
        
    def train_from_mini_batch(self, states, actions, rewards, next_states, is_terminals):
        keras_backend.set_learning_phase(1)
        targets = np.zeros_like(rewards)
        
        stub_cell_input = [np.zeros([len(states),32])]*6
        stub_cell_output = stub_cell_input[:4]
        next_states = [state[-1] for state in next_states]
        states = [state[-1] for state in states]
        next_state_predicted_rewards = self.model.predict([next_states] + stub_cell_input)
        next_state_predicted_rewards = np.amax(next_state_predicted_rewards[0], axis=1).reshape((-1,))
        
        for index, (reward, is_terminal, predicted_next_reward) in enumerate(zip(rewards, is_terminals, next_state_predicted_rewards)):
            if not is_terminal:
                targets[index] = reward + self.gamma * predicted_next_reward
                assert(not np.any(np.isnan(targets)))
            else:
                targets[index] = reward
                
        targets_f = self.model.predict([states] + stub_cell_input)[0]
        for index, action in enumerate(actions):
            targets_f[index][action] = targets[index]
            
        custom_summary_value_dict = {name: getter() for name, getter in self._custom_summaries.items()}
        self._reporting_callback.set_custom_summary_values({**{
            SummaryNames.LEARNING_RATE : self.learning_rate,
            SummaryNames.CHOICE_CONFIDENCE : self.action_probability_sharpening,
            SummaryNames.AVERAGE_SCORE : self.calculate_average_score(),
        }, **custom_summary_value_dict})
        self.model.fit([states] + stub_cell_input, [targets_f] + stub_cell_output, validation_split=0.25, batch_size = BATCH_SIZE, initial_epoch = self.epoch_counter, epochs=self.epoch_counter + EPOCHS_PER_TRAIN_STEP, callbacks=[self.tensorboard, self.checkpointer, self._reporting_callback], shuffle = False)
        self.epoch_counter = self.epoch_counter + EPOCHS_PER_TRAIN_STEP
        
        keras_backend.set_learning_phase(0)

    def increment_probability_sharpening(self):
        self.action_probability_sharpening = min(self.action_probability_sharpening + self.action_probability_sharpening_increase, self.action_probability_sharpening_max)

    def on_end_episode(self):
        self.model.reset_states()
        self._last_games_rewards.append(self._this_game_reward)
        self._this_game_reward = 0

    def store_action_reward(self, reward):
        self._this_game_reward += reward

    def calculate_average_score(self):
        if len(self._last_games_rewards) == 0:
            return 0
        else:
            return sum(self._last_games_rewards) / len(self._last_games_rewards)
