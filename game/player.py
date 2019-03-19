import pyglet

from . import single_frame_actions

from abc import ABC, abstractmethod

class Player(ABC):
    def gather_player_perceived_world_state(self, world):
        return 1

    @abstractmethod
    def get_this_frame_actions(self, perceived_world_state):
        return single_frame_actions.SingleFrameActions(0, 0, 0)