from math import sin, cos, tan, pi
from random import random
from typing import Tuple
import numpy as np

import Box2D

from . import single_frame_actions, physics_object, player_ship, borders, asteroid_play_space
from .collision_filter_categories import CollisionFilterCategory
from .update_result import UpdateResult
from .contact_damage_inflicter import ContactDamageInflicter

class Borders:
    LEFT, TOP, RIGHT, BOTTOM = range(4)

PLAYER_RADIUS = 2
ASTEROID_RADIUS = 5

CONTACT_IMPULSE_TO_DAMAGE_SCALAR = 0.001
PLAYER_CONTACT_RESISTANCE = 0
ASTEROID_CONTACT_RESISTANCE = 0.5

LEFT_BORDER_X = 0
RIGHT_BORDER_X = 80
BOTTOM_BORDER_Y = 0
TOP_BORDER_Y = 60

class World():
    def __init__(self):
        self._player_maximum_thrust = 2

        self._physics_world = Box2D.b2World(gravity=(0,0), doSleep=False, contactListener=ContactDamageInflicter(self))

        self._asteroid_shape = Box2D.b2CircleShape()
        self._asteroid_shape.radius = ASTEROID_RADIUS

        self._asteroid_fixture = Box2D.b2FixtureDef()
        self._asteroid_fixture.shape = self._asteroid_shape
        self._asteroid_fixture.density = 10
        self._asteroid_fixture.restitution = 1
        self._asteroid_fixture.friction = 1
        self._asteroid_fixture.filter.categoryBits = CollisionFilterCategory.ASTEROID
        self._asteroid_fixture.filter.maskBits = CollisionFilterCategory.ASTEROID | CollisionFilterCategory.PLAYER 

        self._ship_shape = Box2D.b2CircleShape()
        self._ship_shape.radius = PLAYER_RADIUS

        ship_fixture = Box2D.b2FixtureDef()
        ship_fixture.shape = self._ship_shape
        ship_fixture.density = 5
        ship_fixture.restitution = 1
        ship_fixture.friction = 1
        ship_fixture.filter.categoryBits = CollisionFilterCategory.PLAYER
        ship_fixture.filter.maskBits = CollisionFilterCategory.ASTEROID | CollisionFilterCategory.BORDER

        self._player_base_friction = 0.02
        self._player_thrust_extra_friction = 0.02

        self._player_maximum_turn_thrust = 1
        self._thrust_extra_turn_thrust = -0.3
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
        self._asteroids_to_kill = []

        self._borders = borders.add_borders(self._physics_world, LEFT_BORDER_X, RIGHT_BORDER_X, BOTTOM_BORDER_Y, TOP_BORDER_Y)
        self._asteroid_play_space = asteroid_play_space.add_asteroid_play_space(self._physics_world, LEFT_BORDER_X, RIGHT_BORDER_X, BOTTOM_BORDER_Y, TOP_BORDER_Y)

        self.player_current_health = 3

    def update(self, player_actions : single_frame_actions.SingleFrameActions):
        self._player_ship.ApplyAngularImpulse(-player_actions.turn_speed * (self._player_maximum_turn_thrust + self._thrust_extra_turn_thrust * player_actions.thrust), True)
        self._player_ship.linearDamping = self._player_base_friction + self._player_thrust_extra_friction * player_actions.thrust

        player_forward_vector = np.array([-sin(self._player_ship.angle), cos(self._player_ship.angle)])
        player_thrust = player_forward_vector * (player_actions.thrust * self._player_maximum_thrust)
        self._player_ship.ApplyLinearImpulse(player_thrust, point=self._player_ship.position, wake=True)
        self._player_ship.angularDamping = self._base_rotational_friction + self._thrust_extra_rotational_friction * player_actions.thrust

        self._physics_world.Step(1, 6, 2)

        asteroids_in_play_space = asteroid_play_space.report_objects_in_play_space(self._asteroid_play_space)

        for asteroid in self._asteroids:
            if asteroid not in asteroids_in_play_space and asteroid not in self._asteroids_to_kill:
                self._asteroids_to_kill.append(asteroid)

        for dead_asteroid in self._asteroids_to_kill:
            self._physics_world.DestroyBody(dead_asteroid)
            self._asteroids.remove(dead_asteroid)
        self._asteroids_to_kill.clear()

        return UpdateResult.CONTINUE_GAME if self.player_current_health > 0 else UpdateResult.GAME_COMPLETED

    def add_player(self, player_controller):
        self._player_controller = player_controller

    def player_impact(self, normal_impulse : float, tangent_impulse : float, impact_asteroid : Box2D.b2Body):
        contact_raw_damage = (normal_impulse ** 2 + tangent_impulse ** 2) * CONTACT_IMPULSE_TO_DAMAGE_SCALAR
        contact_player_damage = max(0, contact_raw_damage - PLAYER_CONTACT_RESISTANCE)
        self.player_current_health = self.player_current_health - contact_player_damage
        if contact_raw_damage > ASTEROID_CONTACT_RESISTANCE and impact_asteroid in self._asteroids:
            self._asteroids_to_kill.append(impact_asteroid)

    @property
    def player(self):
        return self._player_ship

    @property
    def asteroids(self):
        return self._asteroids

    def _create_starting_asteroids(self):
        return [self._create_single_asteroid() for x in range(5)]

    def _find_border_from_center(self, clockwise_angle_from_right : float) -> Tuple[int, Tuple[float, float]]:

        if clockwise_angle_from_right > pi:
            dydx = tan(clockwise_angle_from_right)
            # line: x = my + c
            # m = dydx
            # c = x - my for x:400, y:300
            c = (RIGHT_BORDER_X / 2) - dydx * (TOP_BORDER_Y / 2)

            # resolve upwards
            # target: y = TOP_BORDER_Y
            # x = c + TOP_BORDER_Y * dydx
            x = c + TOP_BORDER_Y * dydx
            if x > LEFT_BORDER_X and x < RIGHT_BORDER_X:
                return (Borders.TOP, (x, TOP_BORDER_Y))
            elif x <= LEFT_BORDER_X:
                # 0 = dydx * y + c
                # -c / dydx = y
                left_wall_intersect = -c / dydx
                return (Borders.LEFT, (LEFT_BORDER_X, left_wall_intersect))
            else:
                # RIGHT_BORDER_X = dydx * y + c
                # (RIGHT_BORDER_X - c) / dydx = y
                right_wall_intersect = (RIGHT_BORDER_X - c) / dydx
                return (Borders.RIGHT, (RIGHT_BORDER_X, right_wall_intersect))
        elif clockwise_angle_from_right < pi and clockwise_angle_from_right > 0:
            dydx = tan(clockwise_angle_from_right)
            # line: x = my + c
            # m = dydx
            # c = x - my for x:400, y:300
            c = (RIGHT_BORDER_X / 2) - dydx * (TOP_BORDER_Y / 2)

            # resolve downwards
            # target: y = 0 
            # x = c
            if c > LEFT_BORDER_X and c < RIGHT_BORDER_X:
                return (Borders.BOTTOM, (c, BOTTOM_BORDER_Y))
            elif c <= LEFT_BORDER_X:
                # 0 = dydx * y + c
                # -c / dydx = y
                left_wall_intersect = -c / dydx
                return (Borders.LEFT, (LEFT_BORDER_X, left_wall_intersect))
            else:
                # RIGHT_BORDER_X = dydx * y + c
                # (RIGHT_BORDER_X - c) / dydx = y
                right_wall_intersect = (RIGHT_BORDER_X - c) / dydx
                return (Borders.RIGHT, (RIGHT_BORDER_X, right_wall_intersect))
        elif clockwise_angle_from_right is 0:
            return (Borders.RIGHT, (RIGHT_BORDER_X, TOP_BORDER_Y / 2))
        else:
            return (Borders.LEFT, (LEFT_BORDER_X, TOP_BORDER_Y / 2))

    def _create_single_asteroid(self):
        direction = random() * 2 * pi

        border_collided_with, collision_point = self._find_border_from_center(direction)
        
        # place asteroid beyond border
        border_offsets = {
            Borders.LEFT: (-ASTEROID_RADIUS, 0),
            Borders.TOP: (0, ASTEROID_RADIUS),
            Borders.RIGHT: (ASTEROID_RADIUS, 0),
            Borders.BOTTOM: (0, -ASTEROID_RADIUS)
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