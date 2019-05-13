import pyglet
from pyglet import clock
import math
from typing import Callable, Tuple
import Box2D

from ..game.world import World, MAX_PLAYER_HEALTH
from ..game.update_result import UpdateResult
from . import background, resources

window_dimensions = {'x':800, 'y':600}

def load_background(background_batch, window_dimensions):
    return background.load_background(background_batch, window_dimensions)

def load_player_sprite(player_batch):
    return pyglet.sprite.Sprite(img=resources.player, batch=player_batch)

def load_player_damage_sprites(player_batch, _player_sprite):
    damage_sprites = []
    for sprite_resource in resources.player_damage:
        damage_sprite = pyglet.sprite.Sprite(img=sprite_resource, batch=player_batch)
        damage_sprite.visible = False
        damage_sprites.append(damage_sprite)
    return damage_sprites

MAX_UNCONSUMED_TIME = 0.05

class Renderer():
    def __init__(self, update_callback : Callable[[], UpdateResult]= None, get_world_callback : Callable[[], World] = None):
        self._game_window = pyglet.window.Window(window_dimensions['x'], window_dimensions['y'])

        self._magnification = 10

        self._background_batch = pyglet.graphics.Batch()
        self._player_batch = pyglet.graphics.Batch()
        self._asteroid_batch = pyglet.graphics.Batch()

        self._background_sprites = load_background(self._background_batch, window_dimensions)
        self._player_sprite = load_player_sprite(self._player_batch)
        self._player_damage_sprites = load_player_damage_sprites(self._player_batch, self._player_sprite)
        self._asteroid_sprites = []
        
        game_framerate = 1/120

        self.unconsumed_time = 0
        def update(dt):
            self.unconsumed_time = min(self.unconsumed_time + dt, MAX_UNCONSUMED_TIME)
            frames_to_consume = math.floor(self.unconsumed_time / game_framerate)
            for _ in range(frames_to_consume):
                update_result = update_callback()
                if update_result is UpdateResult.GAME_COMPLETED:
                    self._game_window.close()
            self.unconsumed_time = self.unconsumed_time - frames_to_consume * game_framerate
        
        pyglet.clock.schedule_interval(update, game_framerate)

        self._get_world = get_world_callback

        def on_draw():
            self._game_window.clear()

            self._apply_world_to_render_state(self._get_world())

            self._background_batch.draw()
            self._player_batch.draw()
            self._asteroid_batch.draw()
        self._game_window.event(on_draw)

    def _apply_coordinates_to_sprite(self, sprite : pyglet.sprite.Sprite, body : Box2D.b2Body):
        sprite.x, sprite.y = body.position * self._magnification
        sprite.rotation = -math.degrees(body.angle)

    @property
    def game_window(self):
        return self._game_window

    def run(self):
        pyglet.app.run()

    def _apply_world_to_render_state(self, world):

        for sprite in [self._player_sprite] + self._player_damage_sprites:
            self._apply_coordinates_to_sprite(sprite, world.player)
        self._apply_damage_state_decals(world.player_current_health)

        difference_in_number_of_asteroids = len(world.asteroids) - len(self._asteroid_sprites)
        if difference_in_number_of_asteroids > 0:
            self._asteroid_sprites.extend([pyglet.sprite.Sprite(img=resources.asteroid, batch=self._asteroid_batch) for missing_asteroid_index in range(difference_in_number_of_asteroids)])
        elif difference_in_number_of_asteroids < 0:
            for deleting_asteroid in self._asteroid_sprites[:-difference_in_number_of_asteroids]:
                deleting_asteroid.delete()
            self._asteroid_sprites = self._asteroid_sprites[-difference_in_number_of_asteroids:]
        
        for asteroid_physics_body, asteroid_sprite in zip(world.asteroids, self._asteroid_sprites):
            self._apply_coordinates_to_sprite(asteroid_sprite, asteroid_physics_body)

    def _apply_damage_state_decals(self, current_health):
        sprite_index_to_enable = None

        health_proportion = current_health / MAX_PLAYER_HEALTH

        if health_proportion <= 1/4:
            sprite_index_to_enable = 2
        elif health_proportion < 2/4:
            sprite_index_to_enable = 1
        elif health_proportion < 3/4:
            sprite_index_to_enable = 0

        for sprite_index, sprite in enumerate(self._player_damage_sprites):
            sprite.visible = sprite_index == sprite_index_to_enable
