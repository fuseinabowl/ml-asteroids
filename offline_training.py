import pyglet

import argparse
import os.path
import glob
import datetime

from collections import deque
from random import shuffle

from asteroids import env as environment
from asteroids.renderer import renderer
from asteroids.game import update_result
from asteroids.agents import dqn_agent

from tensorflow.keras.models import load_model

import numpy as np

NUMBER_OF_REPLAY_FRAMES_STORED = 20000
TRAINING_PERIOD = 2000
PRIORITY_EPSILON = 100

class ReplayFrame():
    def __init__(
        self,
        observation,
        action,
        reward,
        next_observation,
        is_done
    ):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.is_done = is_done
        self.priority = None # to be filled by caller later

class OfflineTraining():

    def main(self, custom_model = None):
        self.env = environment.Env()
        self.replays = deque(maxlen = NUMBER_OF_REPLAY_FRAMES_STORED)
        self.last_seen_observation = self.env.reset()
        self.agent = dqn_agent.DQNAgent(self.env.observation_space.shape[0], self.env.action_space.spaces[0].n, model=custom_model)

        self.steps_completed = 0
        self.training_period = TRAINING_PERIOD

        self._last_action_estimated_quality = None
        self._previous_frame_replay_data = None
        
        def update_game():
            player_actions_as_single_value, action_quality, best_action_quality = self.agent.act(self.last_seen_observation)
            next_observation, reward, is_done, _ = self.env.step(player_actions_as_single_value)
            if self._previous_frame_replay_data:
                assert(self._last_action_estimated_quality)
                future_quality_contribution = (self.agent.gamma * best_action_quality) if not self._previous_frame_replay_data.is_done else 0
                last_action_quality_with_estimated_future = self._previous_frame_replay_data.reward + future_quality_contribution
                self._previous_frame_replay_data.priority = abs(last_action_quality_with_estimated_future - self._previous_frame_replay_data.reward) + PRIORITY_EPSILON
                self.replays.append(self._previous_frame_replay_data)
                
            self._previous_frame_replay_data = ReplayFrame(self.last_seen_observation, player_actions_as_single_value, reward, next_observation, is_done)
                
            self._last_action_estimated_quality = action_quality
            self.last_seen_observation = next_observation
            self.agent.store_action_reward(reward)

            self.steps_completed = self.steps_completed + 1
            if self.steps_completed % self.training_period == 0:
                mini_batch = self.generate_mini_batch()
                observations = [frame.observation for frame in mini_batch]
                actions = [frame.action for frame in mini_batch]
                rewards = [frame.reward for frame in mini_batch]
                next_observations = [frame.next_observation for frame in mini_batch]
                is_dones = [frame.is_done for frame in mini_batch]
                self.agent.train_from_mini_batch(observations, actions, rewards, next_observations, is_dones)

            if is_done:
                self.last_seen_observation = self.env.reset()
                self.agent.on_end_episode()

            return update_result.UpdateResult.CONTINUE_GAME

        def get_world():
            return self.env.world

        renderer_instance = renderer.Renderer(update_game, get_world)
        
        renderer_instance.run()

    def generate_mini_batch(self):
        batch_data = []
        for index in range(dqn_agent.TIMESPAN_LENGTH, len(self.replays)):
            this_frame = self.replays[index]
            this_frame_historic_observations = np.concatenate([self.replays[replay_frame_index].observation for replay_frame_index in range(index - dqn_agent.TIMESPAN_LENGTH, index)], axis=0)
            this_frame_next_observation_with_history = np.concatenate([self.replays[replay_frame_index].next_observation for replay_frame_index in range(index - dqn_agent.TIMESPAN_LENGTH, index)], axis=0)
            batch_data.append(ReplayFrame(
                observation = this_frame_historic_observations,
                action = this_frame.action,
                reward = this_frame.reward,
                next_observation = this_frame_next_observation_with_history,
                is_done = this_frame.is_done
            ))
        return batch_data

def get_latest_file(file_pattern,path=None):
    if path is None:
        list_of_files = glob.glob('{0}'.format(file_pattern))
        if len(list_of_files)> 0:
            return os.path.split(max(list_of_files, key = os.path.getctime))[1]
    else:
        list_of_files = glob.glob('{0}/{1}'.format(path, file_pattern))
        if len(list_of_files) > 0:
            return os.path.split(max(list_of_files,key=os.path.getctime))[1]
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an agent to play the asteroids gym.')
    parser.add_argument('--resume_last_training_session', action='store_true')
    args = parser.parse_args()
    
    model = None

    if args.resume_last_training_session:
        model_filepath = get_latest_file(file_pattern='*.model', path='models')
        if model_filepath:
            model = load_model(os.path.join('models', model_filepath))

    trainer = OfflineTraining()
    trainer.main(model)