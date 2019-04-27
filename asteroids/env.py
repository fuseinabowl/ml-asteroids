import gym

from gym.spaces import Tuple, Discrete, Box
from .game import world, agent_player
from .game.update_result import UpdateResult

OBSERVATION_SPACE_PROXIMITY_MAXIMUM_DISTANCE = 100
OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS = 20

class Env(gym.Env):
    action_space = None
    observation_space = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = Tuple(
            [Discrete(3) # turn left, don't turn, turn right
            ,Discrete(2) # don't thrust, thrust
            ]
        )
        # proximity sensor, 0 is touching 100 is far.
        # Index 0 is at the ship's nose, further indices move
        # clockwise around (index n-1 will be one counter clockwise from ship's nose)
        self.observation_space = Box(0, OBSERVATION_SPACE_PROXIMITY_MAXIMUM_DISTANCE, shape=(OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS, 1))

        self.reset()

    def step(self, action):
        game_actions = agent_player.convert_gym_actions_to_world_actions(action)

        update_result = self._world.update(game_actions)
        is_env_done = update_result == UpdateResult.GAME_COMPLETED
        observation = self._gather_player_perceived_world_state()
        return observation, self._world.player_current_health, is_env_done

    def reset(self):
        self._world = world.World()
        return self._gather_player_perceived_world_state()

    def _gather_player_perceived_world_state(self):
        return self._world.do_proximity_raycasts_from_player(OBSERVATION_SPACE_PROXIMITY_NUMBER_OF_RAYS, OBSERVATION_SPACE_PROXIMITY_MAXIMUM_DISTANCE)

    @property
    def world(self):
        return self._world