from math import sin, cos, tan, pi
from random import random
from typing import Tuple
import numpy as np

import Box2D

from . import single_frame_actions, physics_object, player_ship
from .update_result import UpdateResult

class Borders:
    LEFT, TOP, RIGHT, BOTTOM = range(4)

PLAYER_RADIUS = 20
ASTEROID_RADIUS = 50

class World():
    def __init__(self):
        self._player_maximum_thrust = 1

        self._physics_world = Box2D.b2World(gravity=(0,0), doSleep=False)

        self._asteroid_shape = Box2D.b2CircleShape()
        self._asteroid_shape.radius = 5

        self._asteroid_fixture = Box2D.b2FixtureDef()
        self._asteroid_fixture.shape = self._asteroid_shape
        self._asteroid_fixture.density = 10

        self._ship_shape = Box2D.b2CircleShape()
        self._ship_shape.radius = 2

        ship_fixture = Box2D.b2FixtureDef()
        ship_fixture.shape = self._ship_shape
        ship_fixture.density = 5

        self._player_base_friction = 0.02
        self._player_thrust_extra_friction = 0.1

        self._player_maximum_turn_thrust = 0.2
        self._thrust_extra_turn_thrust = -0.1
        self._base_rotational_friction = 0.1
        self._thrust_extra_rotational_friction = 0.05

        ship_body_def = Box2D.b2BodyDef()
        ship_body_def.position = (5, 5)
        ship_body_def.angle = random() * pi * 2
        ship_body_def.linearVelocity = (0,0)
        ship_body_def.linearDamping = self._player_base_friction
        ship_body_def.angularVelocity = 0
        ship_body_def.angularDamping = self._base_rotational_friction
        ship_body_def.fixtures = [ship_fixture]
        ship_body_def.type = Box2D.b2_dynamicBody
        ship_body_def.allowSleep = False

        self._player_ship = self._physics_world.CreateBody(ship_body_def)
        self._player_controller = None

        self._asteroids = self._create_starting_asteroids()

        self.player_current_health = 3

    def update(self, player_actions : single_frame_actions.SingleFrameActions):
        self._player_ship.ApplyAngularImpulse(player_actions.turn_speed * (self._player_maximum_turn_thrust + self._thrust_extra_turn_thrust * player_actions.thrust), True)
        self._player_ship.linearDamping = self._player_base_friction# + self._player_thrust_extra_friction * player_actions.thrust

        print (self._player_ship.mass)

        player_forward_vector = np.array([sin(self._player_ship.angle), cos(self._player_ship.angle)])
        player_thrust = player_forward_vector * (player_actions.thrust * self._player_maximum_thrust)
        self._player_ship.ApplyLinearImpulse(player_thrust, point=self._player_ship.position, wake=True)
        self._player_ship.angularDamping = self._base_rotational_friction + self._thrust_extra_rotational_friction * player_actions.thrust

#        asteroids_to_remove = []
#        for asteroid in self._asteroids:
#            asteroid.update()
#            if self._check_player_collision_with_asteroid(asteroid):
#                self._player_ship.current_health = self._player_ship.current_health - 1
#                asteroids_to_remove.append(asteroid)
#
#        for dead_asteroid in asteroids_to_remove:
#            self._asteroids.remove(dead_asteroid)

        self._physics_world.Step(1, 6, 2)

        return UpdateResult.CONTINUE_GAME if self.player_current_health > 0 else UpdateResult.GAME_COMPLETED

    def add_player(self, player_controller):
        self._player_controller = player_controller

    @property
    def player(self):
        return self._player_ship

    @property
    def asteroids(self):
        return self._asteroids

    def _create_starting_asteroids(self):
        return [self._create_single_asteroid() for x in range(5)]

    def _find_border_from_center(self, clockwise_angle_from_right : float) -> Tuple[int, Tuple[float, float]]:
        left_border_x = 0
        right_border_x = 80
        bottom_border_y = 0
        top_border_y = 60

        if clockwise_angle_from_right > pi:
            dydx = tan(clockwise_angle_from_right)
            # line: x = my + c
            # m = dydx
            # c = x - my for x:400, y:300
            c = (right_border_x / 2) - dydx * (top_border_y / 2)

            # resolve upwards
            # target: y = top_border_y
            # x = c + top_border_y * dydx
            x = c + top_border_y * dydx
            if x > left_border_x and x < right_border_x:
                return (Borders.TOP, (x, top_border_y))
            elif x <= left_border_x:
                # 0 = dydx * y + c
                # -c / dydx = y
                left_wall_intersect = -c / dydx
                return (Borders.LEFT, (left_border_x, left_wall_intersect))
            else:
                # right_border_x = dydx * y + c
                # (right_border_x - c) / dydx = y
                right_wall_intersect = (right_border_x - c) / dydx
                return (Borders.RIGHT, (right_border_x, right_wall_intersect))
        elif clockwise_angle_from_right < pi and clockwise_angle_from_right > 0:
            dydx = tan(clockwise_angle_from_right)
            # line: x = my + c
            # m = dydx
            # c = x - my for x:400, y:300
            c = (right_border_x / 2) - dydx * (top_border_y / 2)

            # resolve downwards
            # target: y = 0 
            # x = c
            if c > left_border_x and c < right_border_x:
                return (Borders.BOTTOM, (c, bottom_border_y))
            elif c <= left_border_x:
                # 0 = dydx * y + c
                # -c / dydx = y
                left_wall_intersect = -c / dydx
                return (Borders.LEFT, (left_border_x, left_wall_intersect))
            else:
                # right_border_x = dydx * y + c
                # (right_border_x - c) / dydx = y
                right_wall_intersect = (right_border_x - c) / dydx
                return (Borders.RIGHT, (right_border_x, right_wall_intersect))
        elif clockwise_angle_from_right is 0:
            return (Borders.RIGHT, (right_border_x, top_border_y / 2))
        else:
            return (Borders.LEFT, (left_border_x, top_border_y / 2))

    def _create_single_asteroid(self):
        direction = random() * 2 * pi

        border_collided_with, collision_point = self._find_border_from_center(direction)
        
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
        asteroid_speed = 0.05
        def create_velocity_generator(start_angle, end_angle):
            angle_range = end_angle - start_angle
            def generate_velocity():
                chosen_angle = start_angle + random() * angle_range
                return (cos(chosen_angle) * asteroid_speed, -sin(chosen_angle) * asteroid_speed)
            return generate_velocity
        border_velocity_generators = {
            Borders.LEFT: create_velocity_generator    (-1 * pi / 4, 1 * pi / 4),
            Borders.TOP: create_velocity_generator     ( 1 * pi / 4, 3 * pi / 4),
            Borders.RIGHT: create_velocity_generator   ( 3 * pi / 4, 5 * pi / 4),
            Borders.BOTTOM: create_velocity_generator  ( 5 * pi / 4, 7 * pi / 4)
        }
        velocity = border_velocity_generators[border_collided_with]()

        max_rotational_velocity = 0.01

        asteroid_body_def = Box2D.b2BodyDef()
        asteroid_body_def.position = asteroid_spawn_location
        asteroid_body_def.angle = random() * pi * 2
        asteroid_body_def.linearVelocity = velocity
        asteroid_body_def.linearDamping = 0
        asteroid_body_def.angularVelocity = (random() * 2 - 1) * max_rotational_velocity
        asteroid_body_def.angularDamping = 0
        asteroid_body_def.fixtures = [self._asteroid_fixture]
        asteroid_body_def.type = Box2D.b2_dynamicBody
        asteroid_body_def.allowSleep = False

        new_asteroid = self._physics_world.CreateBody(asteroid_body_def)

        return new_asteroid

    def _check_player_collision_with_asteroid(self, asteroid : physics_object):
        vector_between_objects = self._player_ship.position - asteroid.position
        distance_between_objects = sum(vector_between_objects ** 2)
        return distance_between_objects < (PLAYER_RADIUS + ASTEROID_RADIUS) ** 2