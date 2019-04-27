import gym

from gym.spaces import Tuple, Discrete, Box
from .game import world, agent_player
from .game.update_result import UpdateResult

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
        self.observation_space = Box(0, 100, shape=(20, 1))

        self.reset()

    def step(self, action):
        # TODO: remove this indirection, apply actions directly
        self._player.set_this_frame_actions_from_action_space(action)
        actions = self._player.get_this_frame_actions(None)

        update_result = self._world.update(actions)
        is_env_done = update_result == UpdateResult.GAME_COMPLETED
        observation = self._player.gather_player_perceived_world_state(self._world)
        return observation, self._world.player_current_health, is_env_done

    def reset(self):
        self._world = world.World()
        self._player = agent_player.AgentPlayer()
        self._world.add_player(self._player)
        return self._player.gather_player_perceived_world_state(self._world)

    @property
    def world(self):
        return self._world