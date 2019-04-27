from . import player, single_frame_actions

class AgentPlayer(player.Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._last_action_turn_speed = 0
        self._last_action_thrust = 0

    def gather_player_perceived_world_state(self, world):
        return 1

    def get_this_frame_actions(self, perceived_world_state):
        return single_frame_actions.SingleFrameActions(
            bullets_firing = 0,
            turn_speed = self._last_action_turn_speed,
            thrust = self._last_action_thrust
        )

    def set_this_frame_actions_from_action_space(self, actions):
        assert(len(actions) == 2)
        assert(actions[0] in range(3))
        assert(actions[1] in range(2))

        self._last_action_turn_speed = actions[0] - 1
        self._last_action_thrust = actions[1]