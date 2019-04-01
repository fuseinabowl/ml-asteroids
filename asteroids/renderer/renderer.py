import pyglet
from pyglet import clock
import math
from typing import Callable, Tuple

from ..game.world import World
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

def apply_coordinates_to_sprite(sprite : pyglet.sprite.Sprite, coordinates : Tuple[float, float], rotation_in_radians : float):
    sprite.x = coordinates[0]
    sprite.y = coordinates[1]
    sprite.rotation = math.degrees(rotation_in_radians)

MAX_UNCONSUMED_TIME = 0.05

class Renderer():
    def __init__(self, update_callback : Callable[[], None]= None, get_world_callback : Callable[[], World] = None):
        self._game_window = pyglet.window.Window(window_dimensions['x'], window_dimensions['y'])

        self._background_batch = pyglet.graphics.Batch()
        self._player_batch = pyglet.graphics.Batch()
        self._asteroid_batch = pyglet.graphics.Batch()

        self._background_sprites = load_background(self._background_batch, window_dimensions)
        self._player_sprite = load_player_sprite(self._player_batch)
        self._player_damage_sprites = load_player_damage_sprites(self._player_batch, self._player_sprite)
        self._asteroid_sprites = []
        
        game_framerate = 1/120

        if update_callback is not None:
            self.unconsumed_time = 0
            def update(dt):
                self.unconsumed_time = min(self.unconsumed_time + dt, MAX_UNCONSUMED_TIME)
                frames_to_consume = math.floor(self.unconsumed_time / game_framerate)
                for _ in range(frames_to_consume):
                    update_callback()
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

    @property
    def game_window(self):
        return self._game_window

    def run(self):
        pyglet.app.run()

    def _apply_world_to_render_state(self, world):

        for sprite in [self._player_sprite] + self._player_damage_sprites:
            apply_coordinates_to_sprite(sprite, world.player.position, world.player.rotation)
        self._apply_damage_state_decals(world.player.current_health)

        difference_in_number_of_asteroids = len(world.asteroids) - len(self._asteroid_sprites)
        if difference_in_number_of_asteroids > 0:
            self._asteroid_sprites.extend([pyglet.sprite.Sprite(img=resources.asteroid, batch=self._asteroid_batch) for missing_asteroid_index in range(difference_in_number_of_asteroids)])
        elif difference_in_number_of_asteroids < 0:
            for deleting_asteroid in self._asteroid_sprites[:-difference_in_number_of_asteroids]:
                deleting_asteroid.delete()
            self._asteroid_sprites = self._asteroid_sprites[-difference_in_number_of_asteroids:]
        
        for asteroid_physics_body, asteroid_sprite in zip(world.asteroids, self._asteroid_sprites):
            apply_coordinates_to_sprite(asteroid_sprite, asteroid_physics_body.position, asteroid_physics_body.rotation)

    def _apply_damage_state_decals(self, current_health):
        sprite_index_to_enable = None

        if current_health == 2:
            sprite_index_to_enable = 0
        elif current_health == 1:
            sprite_index_to_enable = 1
        elif current_health <= 0:
            sprite_index_to_enable = 2

        for sprite_index, sprite in enumerate(self._player_damage_sprites):
            sprite.visible = sprite_index == sprite_index_to_enable
