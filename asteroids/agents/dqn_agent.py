import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, LeakyReLU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import leaky_relu, softmax_cross_entropy_with_logits_v2

from tensorflow.keras.callbacks import TensorBoard
import time

NAME = 'asteroids-pilot-{}'.format(int(time.time()))
EPOCHS_PER_TRAIN_STEP = 5

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.action_probability_sharpening = 0
        self.action_probability_sharpening_increase = 0.05
        self.action_probability_sharpening_max = 5
        self.tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
        self.epoch_counter = 0
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_shape=[1,self.state_size], activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(128, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(128, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(128, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(128, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(self.action_size, activation=leaky_relu))

        model.compile(loss=softmax_cross_entropy_with_logits_v2,
                      optimizer=Adam(lr=self.learning_rate))
        return model
        
    def act(self, state):
        act_values = np.nan_to_num(self.model.predict(state.reshape([1,1,self.state_size])))

        assert(not np.any(np.isnan(act_values)))
        act_values = np.clip(act_values, 0, 100)

        act_values_sharpened = np.nan_to_num(np.power(act_values[0][0], self.action_probability_sharpening * np.ones_like(act_values[0][0])))
        act_values_probabilities = act_values_sharpened / np.sum(act_values_sharpened)
        
        action_selector_value = random.random()
        for action_index, action_probability in enumerate(act_values_probabilities):
            action_selector_value = action_selector_value - action_probability
            if action_selector_value <= 0:
                break
            
        return action_index
        
    def train_from_mini_batch(self, states, actions, rewards, next_states, is_terminals):
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
            
        self.model.fit([states], targets_f, validation_split=0.25, batch_size = 256, initial_epoch = self.epoch_counter, epochs=self.epoch_counter + EPOCHS_PER_TRAIN_STEP, callbacks=[self.tensorboard])
        self.epoch_counter = self.epoch_counter + EPOCHS_PER_TRAIN_STEP
        
        self.action_probability_sharpening = min(self.action_probability_sharpening + self.action_probability_sharpening_increase, self.action_probability_sharpening_max)

    def on_end_episode(self):
        self.model.reset_states()