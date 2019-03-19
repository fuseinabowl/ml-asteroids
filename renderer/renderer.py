import pyglet

from . import background

window_dimensions = {'x':800, 'y':600}

def load_background(background_batch, window_dimensions):
    return background.load_background(background_batch, window_dimensions)

class Renderer():
    def __init__(self, update_callback = None, get_world_callback = None):
        self._game_window = pyglet.window.Window(window_dimensions['x'], window_dimensions['y'])

        self._background_batch = pyglet.graphics.Batch()
        self._main_batch = pyglet.graphics.Batch()

        self._background_sprites = load_background(self._background_batch, window_dimensions)
        
        if update_callback is not None:
            def update(dt):
                update_callback()
            
            pyglet.clock.schedule_interval(update, 1 / 120.0)

        self._get_world_callback = get_world_callback

        @self._game_window.event
        def on_draw():
            self._game_window.clear()
            self._background_batch.draw()
            self._main_batch.draw()

    def run(self):
        pyglet.app.run()
