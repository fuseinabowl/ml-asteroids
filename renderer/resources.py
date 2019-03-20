import pyglet

pyglet.resource.path = ['renderer/resources']
pyglet.resource.reindex()

def center_image(image : pyglet.resource.image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width / 2
    image.anchor_y = image.height / 2

background = pyglet.resource.image("purple.png")
player = pyglet.resource.image("player_ship.png")
center_image(player)