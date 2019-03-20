from . import single_frame_actions, physics_object

class World():
    def __init__(self):
        self._player_maximum_thrust = 0.1
        self._player_ship = physics_object.PhysicsObject(x=50,y=50)
        self._player_controller = None

    def update(self, player_actions : single_frame_actions.SingleFrameActions):
        self._player_ship.velocity[1] = self._player_ship.velocity[1] + player_actions.thrust * self._player_maximum_thrust
        self._player_ship.update()

    def add_player(self, player_controller):
        self._player_controller = player_controller

    @property
    def player(self):
        return self._player_ship