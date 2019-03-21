from math import sin, cos, pi
from random import random

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

        self._asteroids = self._create_starting_asteroids()

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

    def _create_starting_asteroids(self):
        return [self._create_single_asteroid() for x in range(5)]

    def _create_single_asteroid(self):
        direction = random() * 2 * pi

        class Borders:
            LEFT, TOP, RIGHT, BOTTOM = range(4)
        border_collided_with = Borders.LEFT
        collision_point = (0,0)
        
        # find border in direction from centre
        left_border_x = 0
        right_border_x = 800
        bottom_border_y = 0
        top_border_y = 600

        if direction > pi:
            pass
        elif direction < pi and direction > 0:
            pass
        elif direction is 0:
            border_collided_with = Borders.RIGHT
            collision_point = (right_border_x, top_border_y / 2)
        else:
            border_collided_with = Borders.LEFT
            collision_point = (0, top_border_y / 2)

        # place asteroid beyond border
        asteroid_width = 50
        border_offsets = {
            Borders.LEFT: (-asteroid_width, 0),
            Borders.TOP: (0, asteroid_width),
            Borders.RIGHT: (asteroid_width, 0),
            Borders.BOTTOM: (0, -asteroid_width)
        }

        border_offset = border_offsets[border_collided_with]

        asteroid_spawn_location = tuple(collision_coordinate + offset_coordinate for collision_coordinate, offset_coordinate in zip(collision_point, border_offset))

        # make asteroids fly in
        asteroid_speed = 1
        def create_velocity_generator(start_angle, end_angle):
            angle_range = end_angle - start_angle
            def generate_velocity():
                chosen_angle = start_angle + random() * angle_range
                return (sin(chosen_angle) * asteroid_speed, cos(chosen_angle) * asteroid_speed)
            return generate_velocity
        border_velocity_generators = {
            Borders.LEFT: create_velocity_generator    (-1 * pi / 2, 1 * pi / 2),
            Borders.TOP: create_velocity_generator     ( 1 * pi / 2, 3 * pi / 2),
            Borders.RIGHT: create_velocity_generator   ( 3 * pi / 2, 5 * pi / 2),
            Borders.BOTTOM: create_velocity_generator  ( 5 * pi / 2, 7 *pi / 2)
        }
        velocity = border_velocity_generators[border_collided_with]()

        new_asteroid = physics_object.PhysicsObject(
            x=asteroid_spawn_location[0], y=asteroid_spawn_location[1],
            x_velocity=velocity[0], y_velocity=velocity[1]
        )
        return new_asteroid