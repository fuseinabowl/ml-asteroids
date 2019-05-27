import pyglet

from collections import deque
from random import sample

from asteroids import env as environment
from asteroids.renderer import renderer
from asteroids.game import update_result
from asteroids.agents import dqn_agent

NUMBER_OF_REPLAY_FRAMES_STORED = 50000

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

class OfflineTraining():

    def main(self):
        self.env = environment.Env()
        self.replays = deque(maxlen = NUMBER_OF_REPLAY_FRAMES_STORED)
        self.last_seen_observation = self.env.reset()
        self.agent = dqn_agent.DQNAgent(self.env.observation_space.shape[0], self.env.action_space.spaces[0].n)

        self.batch_size = 256

        self.games_remaining = 100000

        self.steps_completed = 0
        self.training_period = 256
        
        def update_game():
            player_actions_as_single_value = self.agent.act(self.last_seen_observation)
            next_observation, reward, is_done, _ = self.env.step(player_actions_as_single_value)
            self.replays.append(ReplayFrame(self.last_seen_observation, player_actions_as_single_value, reward, next_observation, is_done))
            self.last_seen_observation = next_observation

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
                self.games_remaining = self.games_remaining - 1
                if self.games_remaining <= 0:
                    return update_result.UpdateResult.GAME_COMPLETED

            return update_result.UpdateResult.CONTINUE_GAME

        def get_world():
            return self.env.world

        renderer_instance = renderer.Renderer(update_game, get_world)
        
        renderer_instance.run()

    def generate_mini_batch(self):
        return sample(self.replays, self.batch_size)


if __name__ == '__main__':
    trainer = OfflineTraining()
    trainer.main()