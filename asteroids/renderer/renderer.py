import pyglet
from pyglet import clock
import math
from typing import Callable

from ..game.world import World
from . import background, resources

window_dimensions = {'x':800, 'y':600}

def load_background(background_batch, window_dimensions):
    return background.load_background(background_batch, window_dimensions)

def load_player_sprite(player_batch):
    return pyglet.sprite.Sprite(img=resources.player, batch=player_batch)

class Renderer():
    def __init__(self, update_callback : Callable[[], None]= None, get_world_callback : Callable[[], World] = None):
        self._game_window = pyglet.window.Window(window_dimensions['x'], window_dimensions['y'])

        self._background_batch = pyglet.graphics.Batch()
        self._player_batch = pyglet.graphics.Batch()
        self._asteroid_batch = pyglet.graphics.Batch()

        self._background_sprites = load_background(self._background_batch, window_dimensions)
        self._player_sprite = load_player_sprite(self._player_batch)
        self._asteroid_sprites = []
        
        game_framerate = 1/120

        if update_callback is not None:
            self.unconsumed_time = 0
            def update(dt):
                self.unconsumed_time = self.unconsumed_time + dt
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
        self._player_sprite.x = world.player.position[0]
        self._player_sprite.y = world.player.position[1]
        self._player_sprite.rotation = math.degrees(world.player.rotation)

        difference_in_number_of_asteroids = len(world.asteroids) - len(self._asteroid_sprites)
        if difference_in_number_of_asteroids > 0:
            self._asteroid_sprites.extend([pyglet.sprite.Sprite(img=resources.asteroid, batch=self._asteroid_batch) for missing_asteroid_index in range(difference_in_number_of_asteroids)])
        elif difference_in_number_of_asteroids < 0:
            for deleting_asteroid in self._asteroid_sprites[-difference_in_number_of_asteroids:-1]:
                deleting_asteroid.delete()
            self._asteroid_sprites = self._asteroid_sprites[0:-difference_in_number_of_asteroids]
        
        for asteroid_physics_body, asteroid_sprite in zip(world.asteroids, self._asteroid_sprites):
            asteroid_sprite.x = asteroid_physics_body.position[0]
            asteroid_sprite.y = asteroid_physics_body.position[1]
            asteroid_sprite.rotation = math.degrees(asteroid_physics_body.rotation)
