import pyglet

def load_background(background_batch, background_asset):
    background_sprite = pyglet.sprite(background_asset, batch=background_batch)