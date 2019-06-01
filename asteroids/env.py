import gym
import numpy as np
from gym.spaces import Tuple, Discrete, Box

from .game import world, agent_player
from .game.update_result import UpdateResult

OBSERVATION_SPACE_PROXIMITY_MAXIMUM_DISTANCE = 100
OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS = 64

BASE_STAY_ALIVE_SCORE = 0.5

class Env(gym.Env):
    action_space = None
    observation_space = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = Tuple(
            [Discrete(6) # turn left, don't turn, turn right
            #,Discrete(2) # don't thrust, thrust
            ]
        )
        # proximity sensor, 0 is touching 100 is far.
        # Index 0 is at the ship's nose, further indices move
        # clockwise around (index n-1 will be one counter clockwise from ship's nose)
        obs_space_low = np.zeros(shape = OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS + 1)
        obs_space_high = np.append(
            (np.ones(shape = OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS)),
            (np.ones(shape = 1))
        )
        self.observation_space = Box(
            low = obs_space_low,
            high = obs_space_high
        )

        self.reset()

    def step(self, action):
        turn_input = action % 3
        thrust_input = action // 3
        game_actions = agent_player.convert_gym_actions_to_world_actions((turn_input, thrust_input))

        update_result = self._world.update(game_actions)
        is_env_done = update_result == UpdateResult.GAME_COMPLETED
        observation = self._gather_player_perceived_world_state()
        score = self.get_score(health = self._world.player_current_health, is_done = is_env_done)
        return observation, score, is_env_done, None

    def reset(self):
        self._world = world.World()
        return self._gather_player_perceived_world_state()

    def _gather_player_perceived_world_state(self):
        proximity_observation = self._world.do_proximity_raycasts_from_player(OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS, OBSERVATION_SPACE_PROXIMITY_MAXIMUM_DISTANCE) / OBSERVATION_SPACE_PROXIMITY_MAXIMUM_DISTANCE
        current_health = self._world.player_current_health
        health_observation = np.array([[np.clip(current_health,0,1)]])
        return np.append(proximity_observation, health_observation, axis=1)

    def get_score(self, health, is_done):
        if is_done:
            return -10000
        else:
            score = BASE_STAY_ALIVE_SCORE
            score = score + (health / world.MAX_PLAYER_HEALTH) / (1 - BASE_STAY_ALIVE_SCORE)
            return score

    @property
    def world(self):
        return self._world