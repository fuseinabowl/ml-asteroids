import pyglet

def load_background(background_batch, background_asset):
    pyglet.sprite.Sprite(img=background_asset, batch=background_batch)