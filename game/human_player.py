import pyglet
from pyglet.window import key

from . import player, single_frame_actions

player_input_keys = {
    'fire' : key.SPACE,
    'turn left' : key.LEFT,
    'turn right' : key.RIGHT,
    'thrust' : key.UP
}

class HumanPlayer(player.Player):
    def __init__(self, pyglet_window, *args, **kwargs):
        super(HumanPlayer, self).__init__(*args, **kwargs)

        self._register_for_events(pyglet_window)

        self._turn_speed = 0
        self._thrust = 0
        self._reset_event_cache()

    def _register_for_events(self, window):
        window.push_handlers(self)

    def _reset_event_cache(self):
        self._fire_requests = 0

    def on_key_press(self, symbol, modifiers):
        if symbol == player_input_keys['fire']:
            self._fire_requests = self._fire_requests + 1

    def get_this_frame_actions(self, perceived_world_state):
        actions = single_frame_actions.SingleFrameActions(
            bullets_firing = self._fire_requests,
            turn_speed = self._turn_speed,
            thrust = self._thrust
        )
        self._reset_event_cache()
        return actions
