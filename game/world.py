from math import sin, cos

from . import single_frame_actions, physics_object

class World():
    def __init__(self):
        self._player_maximum_thrust = 0.3

        self._player_base_friction = 0.01
        self._player_thrust_extra_friction = 0.05

        self._player_maximum_turn_thrust = 0.005
        self._thrust_extra_turn_thrust = -0.001
        self._base_rotational_friction = 0.1
        self._thrust_extra_rotational_friction = 0.1
        self._player_ship = physics_object.PhysicsObject(x=50,y=50)
        self._player_controller = None

    def update(self, player_actions : single_frame_actions.SingleFrameActions):
        self._player_ship.rotational_velocity = self._player_ship.rotational_velocity + player_actions.turn_speed * (self._player_maximum_turn_thrust + self._thrust_extra_turn_thrust * player_actions.thrust)
        self._player_ship.friction = self._player_base_friction + self._player_thrust_extra_friction * player_actions.thrust

        player_forward_vector = [sin(self._player_ship.rotation), cos(self._player_ship.rotation)]
        self._player_ship.velocity = [respective_velocity + player_actions.thrust * self._player_maximum_thrust * respective_player_forward_vector for respective_velocity, respective_player_forward_vector in zip(self._player_ship.velocity, player_forward_vector)]
        self._player_ship.rotational_friction = self._base_rotational_friction + self._thrust_extra_rotational_friction * player_actions.thrust
        self._player_ship.update()

    def add_player(self, player_controller):
        self._player_controller = player_controller

    @property
    def player(self):
        return self._player_ship