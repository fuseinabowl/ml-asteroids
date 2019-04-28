import random
from collections import deque
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Activation
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(256, input_shape=[1, self.state_size], return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape([1,1,20]))
        return np.argmax(act_values[0])  # returns action
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            reshaped_state = state.reshape([1,1,self.state_size])
            reshaped_next_state = next_state.reshape([1,1,self.state_size])

            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(reshaped_next_state)[0])
            target_f = self.model.predict(reshaped_state)
            target_f[0][action] = target
            self.model.fit(reshaped_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay