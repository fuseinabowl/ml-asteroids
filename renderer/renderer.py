import pyglet

from . import background, resources

window_dimensions = {'x':800, 'y':600}

def load_background(background_batch, window_dimensions):
    return background.load_background(background_batch, window_dimensions)

def load_player_sprite(player_batch):
    return pyglet.sprite.Sprite(img=resources.player, batch=player_batch)

class Renderer():
    def __init__(self, update_callback = None, get_world_callback = None):
        self._game_window = pyglet.window.Window(window_dimensions['x'], window_dimensions['y'])

        self._background_batch = pyglet.graphics.Batch()
        self._player_batch = pyglet.graphics.Batch()

        self._background_sprites = load_background(self._background_batch, window_dimensions)
        self._player_sprite = load_player_sprite(self._player_batch)
        
        if update_callback is not None:
            def update(dt):
                update_callback()
            
            pyglet.clock.schedule_interval(update, 1 / 120.0)

        self._get_world = get_world_callback

        def on_draw():
            self._game_window.clear()

            self._apply_world_to_render_state(self._get_world())

            self._background_batch.draw()
            self._player_batch.draw()
        self._game_window.event(on_draw)

    @property
    def game_window(self):
        return self._game_window

    def run(self):
        pyglet.app.run()

    def _apply_world_to_render_state(self, world):
        self._player_sprite.x = world.player[0]
        self._player_sprite.y = world.player[1]
