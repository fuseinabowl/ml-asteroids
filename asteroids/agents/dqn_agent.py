import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, LeakyReLU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import leaky_relu

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.action_probability_sharpening = 0
        self.action_probability_sharpening_increase = 0.05
        self.action_probability_sharpening_max = 5
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_shape=[1,self.state_size], activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(16, activation=leaky_relu))
        model.add(Dropout(rate=0.3))
        model.add(Dense(self.action_size, activation='tanh'))
        #model.add(LeakyReLU(alpha=0.3))

        #model.add(LSTM(64, batch_input_shape=[1,1,self.state_size], return_sequences=True, stateful=True, activation='tanh'))
        #model.add(LeakyReLU(alpha=0.3))
        #model.add(LSTM(64, return_sequences=True, stateful=True, activation='tanh'))
        #model.add(LeakyReLU(alpha=0.3))
        #model.add(LSTM(self.action_size, activation='linear', stateful=True))
        #model.add(LeakyReLU(alpha=0.3))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
        
    def act(self, state):
        act_values = np.nan_to_num(self.model.predict(state.reshape([1,1,self.state_size])))

        assert(not np.isnan(np.sum(act_values)))

        act_values_sharpened = np.power(self.action_probability_sharpening * np.ones_like(act_values[0][0]), act_values[0][0])
        act_values_probabilities = act_values_sharpened / np.sum(act_values_sharpened)
        
        action_selector_value = random.random()
        for action_index, action_probability in enumerate(act_values_probabilities):
            action_selector_value = action_selector_value - action_probability
            if action_selector_value <= 0:
                break
            
        return action_index
        
    def train_one_frame(self, state, action, reward, next_state, done):
        reshaped_state = state.reshape([1,1,self.state_size])
        reshaped_next_state = next_state.reshape([1,1,self.state_size])

        target = reward
        assert(not np.isnan(target))
        if not done:
            target = reward +  \
                    self.gamma * np.amax(np.nan_to_num(self.model.predict(reshaped_next_state)[0]))
            assert(not np.isnan(target))
        target_f = self.model.predict(reshaped_state)
        target_f[0][0][action] = target
        self.model.fit(reshaped_state, target_f, epochs=1, verbose=0)

    def train_from_mini_batch(self, states, actions, rewards, next_states, is_terminals):
        targets = np.zeros_like(rewards)
        
        for index, (reward, is_terminal, next_state) in enumerate(zip(rewards, is_terminals, next_states)):
            if not is_terminal:
                targets[index] = reward +  \
                        self.gamma * np.amax(np.nan_to_num(self.model.predict([next_state.reshape((1,1,-1))])[0]))
                assert(not np.any(np.isnan(targets)))
            else:
                targets[index] = reward
                
        targets_f = self.model.predict([states])
        for index in range(len(targets_f)):
            targets_f[index][0][actions] = targets[index]
            
        self.model.fit([states], targets_f, epochs=1, verbose=0)
        
    def on_end_episode(self):
        self.action_probability_sharpening = min(self.action_probability_sharpening + self.action_probability_sharpening_increase, self.action_probability_sharpening_max)

        self.model.reset_states()