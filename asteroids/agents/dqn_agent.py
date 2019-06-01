import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, LeakyReLU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import leaky_relu
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.losses import mean_squared_error

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time

EPOCHS_PER_TRAIN_STEP = 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.999    # discount rate
        self.learning_rate = 0.001
        self.action_probability_sharpening = 1
        self.action_probability_sharpening_increase = 0.01
        self.action_probability_sharpening_max = 5
        self.epoch_counter = 0
        self.model = self._build_model()

        keras_backend.set_learning_phase(0)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_shape=[1,self.state_size], activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(512, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(512, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(self.action_size, activation=leaky_relu))

        model.compile(loss=mean_squared_error,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    @staticmethod
    def sharpen(mantissa, exponents):
        """ like softmax, but with a custom mantissa """
        mantissa_raised_to_exponents = np.power(mantissa * np.ones_like(exponents), exponents - np.max(exponents))
        return mantissa_raised_to_exponents / mantissa_raised_to_exponents.sum()
        
    def act(self, state):
        act_values = np.nan_to_num(self.model.predict(state.reshape([1,1,self.state_size])))

        assert(not np.any(np.isnan(act_values)))

        act_values_probabilities = DQNAgent.sharpen(self.action_probability_sharpening, act_values[0][0])
        
        action_selector_value = random.random()
        for action_index, action_probability in enumerate(act_values_probabilities):
            action_selector_value = action_selector_value - action_probability
            if action_selector_value <= 0:
                break

        assert(action_selector_value <= 0)
            
        return action_index
        
    def train_from_mini_batch(self, states, actions, rewards, next_states, is_terminals):
        keras_backend.set_learning_phase(1)
        targets = np.zeros_like(rewards)
        
        next_state_predicted_rewards = np.amax(np.nan_to_num(self.model.predict([next_states])), axis=2)
        next_state_predicted_rewards = next_state_predicted_rewards.reshape((-1,))
        
        for index, (reward, is_terminal, predicted_next_reward) in enumerate(zip(rewards, is_terminals, next_state_predicted_rewards)):
            if not is_terminal:
                targets[index] = reward + self.gamma * predicted_next_reward
                assert(not np.any(np.isnan(targets)))
            else:
                targets[index] = reward
                
        targets_f = self.model.predict([states])
        for index, action in enumerate(actions):
            targets_f[index][0][action] = targets[index]
            
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
        checkpointer = ModelCheckpoint(filepath='models/{}.model'.format(NAME), verbose=1, save_best_only=True)
        self.model.fit([states], targets_f, validation_split=0.25, batch_size = 256, initial_epoch = self.epoch_counter, epochs=self.epoch_counter + EPOCHS_PER_TRAIN_STEP, callbacks=[tensorboard, checkpointer])
        self.epoch_counter = self.epoch_counter + EPOCHS_PER_TRAIN_STEP
        
        self.action_probability_sharpening = min(self.action_probability_sharpening + self.action_probability_sharpening_increase, self.action_probability_sharpening_max)
        keras_backend.set_learning_phase(0)

    def on_end_episode(self):
        self.model.reset_states()