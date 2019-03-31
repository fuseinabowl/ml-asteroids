import pyglet

from . import resources

def load_background(background_batch, window_dimensions):
    background_asset = resources.background
    background_sprites = []
    num_tiles_x = window_dimensions['x'] // background_asset.width + 1
    num_tiles_y = window_dimensions['y'] // background_asset.height + 1
    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            background_sprites.append(pyglet.sprite.Sprite(x=x * background_asset.width, y=y * background_asset.height, img=background_asset, batch=background_batch))

    return background_sprites