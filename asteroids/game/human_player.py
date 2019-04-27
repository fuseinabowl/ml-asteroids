import pyglet
from pyglet.window import key

from . import single_frame_actions

player_input_keys = {
    'fire' : key.SPACE,
    'turn left' : key.A,
    'turn right' : key.D,
    'thrust' : key.W
}

class HumanPlayer():
    def __init__(self, pyglet_window, *args, **kwargs):
        self._register_for_events(pyglet_window)

        self._left_pressed = 0
        self._right_pressed = 0
        self._thrust = 0
        self._reset_event_cache()

    def _register_for_events(self, window):
        window.push_handlers(self)

    def _reset_event_cache(self):
        self._fire_requests = 0

    def on_key_press(self, symbol, modifiers):
        if symbol == player_input_keys['fire']:
            self._fire_requests = self._fire_requests + 1
        elif symbol == player_input_keys['turn left']:
            self._left_pressed = 1
        elif symbol == player_input_keys['turn right']:
            self._right_pressed = 1
        elif symbol == player_input_keys['thrust']:
            self._thrust = 1

    def on_key_release(self, symbol, modifiers):
        if symbol == player_input_keys['turn left']:
            self._left_pressed = 0
        elif symbol == player_input_keys['turn right']:
            self._right_pressed = 0
        elif symbol == player_input_keys['thrust']:
            self._thrust = 0

    def _calculate_turn_speed(self):
        return -self._left_pressed + self._right_pressed

    def get_this_frame_actions(self):
        actions = single_frame_actions.SingleFrameActions(
            bullets_firing = self._fire_requests,
            turn_speed = self._calculate_turn_speed(),
            thrust = self._thrust
        )
        self._reset_event_cache()
        return actions
