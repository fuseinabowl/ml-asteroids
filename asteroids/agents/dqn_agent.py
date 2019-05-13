import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

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
        model.add(LSTM(64, batch_input_shape=[1,1,self.state_size], return_sequences=True, stateful=True))
        model.add(LSTM(64, return_sequences=True, stateful=True))
        model.add(LSTM(self.action_size, activation='linear', stateful=True))
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
        
    def on_end_episode(self):
        self.action_probability_sharpening = min(self.action_probability_sharpening + self.action_probability_sharpening_increase, self.action_probability_sharpening_max)

        self.model.reset_states()