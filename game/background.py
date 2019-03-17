import pyglet

def load_background(background_batch, background_asset):
    return pyglet.sprite.Sprite(x=0, y=0, img=background_asset, batch=background_batch)